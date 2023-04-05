import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re

import ktrain
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


from ktrain import text
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from domain_name_extractor import extract_domain_name
from datetime import datetime
import wordsegment as ws
ws.load()

def clean_text(text):
    """
        text: a string

        return: modified initial string
    """
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))

    text = BeautifulSoup(text, "lxml").text  # HTML decoding
    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub(' ', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # delete stopwors from text
    text = ' '.join([i for i in text.split() if not i.isdigit()])
    new = text.replace("_", " ")
    return new


def trainClassifiers(dataset, parent_class):
    dataset = dataset.drop(dataset.index[dataset.name_t.str.contains(r'[0-9]', na=False)])
    dataset = dataset[dataset['name_t'].notnull()]

    # dataset['description'] = dataset['description'].apply(clean_text)

    # parse the domain name
    domains = pd.unique(dataset['page_domain'])
    domains_dict = {}
    for d in domains:
        v = extract_domain_name(d)
        domains_dict[d] = v

    dataset.replace({"page_domain": domains_dict})

    # add the name + domain column
    dataset['domain1'] = dataset['name_t'] + ' ' + dataset['page_domain']

    # add the name + 1 scentence _ domain column
    dataset['domain2'] = dataset['description'] + ' ' + dataset['page_domain']

    # this is the column to be used to train the model
    # 'name_t' for name only
    # 'description' for name + 1 scenternce of description
    # 'domain1' for name and page domain (parsed)
    # 'domain2' for name  1 scenternce of description and page domain (parsed)

    Training_Column = 'domain2'

    possible_labels = dataset.schemaorg_class.unique()

    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index

    # dataset['label'] = dataset.schemaorg_class.replace(label_dict)

    X_train, X_val, y_train, y_val = train_test_split(dataset.index.values,
                                                      dataset.label.values,
                                                      test_size=0.15,
                                                      random_state=42,
                                                      stratify=dataset.label.values)

    data_train, data_test = np.split(dataset.sample(frac=1, random_state=42),
                                     [int(.7 * len(dataset))])

    train_data, test_data, preproc = text.texts_from_df(train_df=data_train,
                                                        text_column=Training_Column,
                                                        label_columns='schemaorg_class',
                                                        val_df=data_test,
                                                        maxlen=64,
                                                        preprocess_mode='distilbert')

    model = text.text_classifier(name='distilbert',
                                 train_data=train_data,
                                 preproc=preproc)

    learner = ktrain.get_learner(model=model, train_data=train_data,
                                 val_data=test_data,
                                 batch_size=16)

    # Essentially fit is a very basic training loop, whereas fit one cycle uses the one cycle policy callback

    learner.fit_onecycle(lr=5e-5, epochs=3)  # 5e-5
    # learner.fit(lr = 0.001 ,n_cycles=10)

    learner.validate(class_names=preproc.get_classes())

    predictor = ktrain.get_predictor(learner.model, preproc)

    predected_class_list=[]
    Instance_Names = parent_class[Training_Column]
    predictions = predictor.predict(list(Instance_Names))
    p = np.argmax(predictions, axis=-1)
    for name in predictions:
        predected_class_list.append(name)

    parent_class['Predected_Class'] = predected_class_list

    filename = Training_Column + '_' + datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") + '.csv'

    parent_class.to_csv(filename)

    return predictor, label_dict


# this method performs the 5-fold training using the same BERT model
def transformer_cv(dataset):
    dataset = dataset.drop(dataset.index[dataset.name_t.str.contains(r'[0-9]', na=False)])
    dataset = dataset[dataset['name_t'].notnull()]

    dataset['description'] = dataset['description'].apply(clean_text)

    # parse the domain name
    dataset['page_domain'] = dataset['page_domain'].apply(extract_domain_name)

    # add the name + domain column
    dataset['domain1'] = dataset['name_t'] + ' ' + dataset['page_domain']

    # add the name + 1 scentence _ domain column
    dataset['domain2'] = dataset['description'] + ' ' + dataset['page_domain']

    # this is the column to be used to train the model
    # 'name_t' for name only
    # 'description' for name + 1 scenternce of description
    # 'domain1' for name and page domain (parsed)
    # 'domain2' for name  1 scenternce of description and page domain (parsed)

    Training_Column = 'domain1'

    possible_labels = dataset.schemaorg_class.unique()

    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index

    # dataset['label'] = dataset.schemaorg_class.replace(label_dict)

    # CV with transformers
    N_FOLDS = 5
    EPOCHS = 3
    LR = 5e-5
    MODEL_NAME = 'distilbert-base-uncased'  # 'bert'

    predictions, accs = [], []
    data = dataset[[Training_Column, 'schemaorg_class']]
    Folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for train_index, val_index in Folds.split(data, data['schemaorg_class']):
        preproc = text.Transformer(MODEL_NAME, maxlen=10)
        train, val = dataset.iloc[train_index], dataset.iloc[val_index]
        x_train = train.name_t.values
        x_val = val.name_t.values

        y_train = train.schemaorg_class.values
        y_val = val.schemaorg_class.values

        trn = preproc.preprocess_train(x_train, y_train)
        model = preproc.get_classifier()
        learner = ktrain.get_learner(model, train_data=trn, batch_size=120)
        learner.fit_onecycle(LR, EPOCHS)
        predictor = ktrain.get_predictor(learner.model, preproc)
        pred = predictor.predict(x_val)
        acc = accuracy_score(y_val, pred)
        print('acc', acc)
        accs.append(acc)

    learner.validate(class_names=preproc.get_classes())

    # predictor = ktrain.get_predictor(learner.model, preproc)
    return predictor, label_dict



##############################################
#
#Place
#
##############################################

dataset=pd.read_csv('/data/lip18oaf/wdc_data_v4/Place.csv')
parent=pd.read_csv('/data/lip18oaf/wdc_data_v4/Place_parent.csv')

original=len(dataset)
dataset.drop_duplicates(subset=['schemaorg_class','name_t','page_domain'], keep='first', inplace=True, ignore_index=True)
new=len(dataset)
print('Total number of duplicate items is', original-new)


Model,label_dict=trainClassifiers(dataset,parent)



##############################################
#
# Local Business
#
##############################################

dataset=pd.read_csv('/data/lip18oaf/wdc_data_v4/LocalBusiness.csv')
parent=pd.read_csv('/data/lip18oaf/wdc_data_v4/LocalBusiness_parent.csv')

original=len(dataset)
dataset.drop_duplicates(subset=['schemaorg_class','name_t','page_domain'], keep='first', inplace=True, ignore_index=True)
new=len(dataset)
print('Total number of duplicate items is', original-new)


Model,label_dict=trainClassifiers(dataset,parent)


##############################################
#
# Creative Work
#
##############################################


dataset=pd.read_csv('/data/lip18oaf/wdc_data_v4/CreativeWork.csv')
parent=pd.read_csv('/data/lip18oaf/wdc_data_v4/CreativeWork_parent.csv')


original=len(dataset)
dataset.drop_duplicates(subset=['schemaorg_class','name_t','page_domain'], keep='first', inplace=True, ignore_index=True)
new=len(dataset)
print('Total number of duplicate items is', original-new)


Model,label_dict=trainClassifiers(dataset,parent)
