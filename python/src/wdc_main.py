import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from bs4 import BeautifulSoup
import re

from collections import Counter

import tensorflow_hub as hub
import ktrain
from ktrain import text
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


from ktrain import text
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


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


def trainClassifiers(dataset):

    dataset=dataset.drop(dataset.index[dataset.name_t.str.contains(r'[0-9]', na=False)])
    dataset = dataset[dataset['name_t'].notnull()]

    #replace null descriptions with ' '  
    dataset["description_t"].fillna("   ", inplace = True)

    #add a sentence of the disc to the name column
    dataset['description'] = dataset['description_t'].apply(lambda x: x.split('.')[0])  
    dataset['name_t'] = dataset['name_t'] + ' ' +  dataset['description']

    dataset['description_t']=dataset['description_t'].apply(clean_text)
    #This because when adding the description class TVstation is cosing some problem (none of the testing data is being classified as TVstaiton)

    dataset = dataset.drop(dataset[dataset['schemaorg_class']=='TelevisionStation'].index)
    #dataset['name_t']=dataset['name_t'].apply(clean_text)

    #removes classes with few instances because BERT is not able to learn from those
    minimum_instances=2
    try:
          stat=dataset.groupby('schemaorg_class').agg({'name_t':'count'})
          MyList=[]
          MyList=stat[stat['name_t'] < minimum_instances].index.values
          dataset=dataset[~dataset.schemaorg_class.isin(MyList)]
    except KeyError:
          print('')
  
   

    possible_labels = dataset.schemaorg_class.unique()

    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index

    dataset['label'] = dataset.schemaorg_class.replace(label_dict)


    X_train, X_val, y_train, y_val = train_test_split(dataset.index.values,
                                                      dataset.label.values,
                                                      test_size=0.15,
                                                      random_state=42,
                                                      stratify=dataset.label.values)

    
    data_train, data_test = np.split(dataset.sample(frac=1, random_state=42),
                                      [int(.7*len(dataset))])


    #data_train['description'] = data_train['description_t'].apply(lambda x: x.split('.')[0])
    #data_train['name_t'] = data_train['name_t'] + ' ' +  data_train['description']


    #data_test['description'] = data_test['description_t'].apply(lambda x: x.split('.')[0])
    #data_test['name_t'] = data_test['name_t'] + ' ' +  data_test['description']



    train_data, test_data, preproc = text.texts_from_df(train_df=data_train,
                                                                        text_column = 'name_t',
                                                                        label_columns = 'schemaorg_class',
                                                                        val_df = data_test,
                                                                        maxlen = 15,
                                                                        preprocess_mode = 'distilbert')

    model = text.text_classifier(name = 'distilbert',
                                  train_data=train_data,
                                  preproc = preproc)

    learner = ktrain.get_learner(model=model, train_data=train_data,
                                  val_data = test_data,
                                  batch_size = 16)

    #Essentially fit is a very basic training loop, whereas fit one cycle uses the one cycle policy callback

    learner.fit_onecycle(lr=5e-5 , epochs = 3)#5e-5
    #learner.fit(lr = 0.001 ,n_cycles=10)
    
    learner.validate(class_names=preproc.get_classes())

    predictor = ktrain.get_predictor(learner.model, preproc)
    #predictor.save('/content/drive/My Drive/bert')
    #y_pred=model.predict(X_test)
    
    #print(classification_report(y_test.argmax(axis=-1), y_pred.argmax(axis=-1)))
    #print(classification_report(y_test.argmax(axis=-1), y_pred.argmax(axis=-1),target_names=possible_labels))

    #plot_classification_report(classificationReport)
      #= classification_report(y_test, y_pred, )

    #R1.to_csv('report.csv')
    return predictor, label_dict


#this method performs the 5-fold training using the same BERT model
def transformer_cv(dataset):

    possible_labels = dataset.schemaorg_class.unique()

    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
            label_dict[possible_label] = index

    dataset['label'] = dataset.schemaorg_class.replace(label_dict)


    x_train, x_val, y_train, y_val = train_test_split(dataset.index.values,
                                                          dataset.label.values,
                                                          test_size=0.15,
                                                          random_state=42,
                                                          stratify=dataset.label.values) 
    
                                                            
    # CV with transformers
    N_FOLDS = 5
    EPOCHS = 3
    LR = 5e-5
    MODEL_NAME='distilbert-base-uncased'#'bert'

    predictions,accs=[],[]
    data = dataset[['name_t', 'schemaorg_class']]
    Folds=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for train_index, val_index in Folds.split(data,data['schemaorg_class']):
        preproc  = text.Transformer(MODEL_NAME, maxlen=10)
        train,val=dataset.iloc[train_index],dataset.iloc[val_index]
        x_train=train.name_t.values
        x_val=val.name_t.values

        y_train=train.schemaorg_class.values
        y_val=val.schemaorg_class.values

        trn = preproc.preprocess_train(x_train, y_train)
        model = preproc.get_classifier()
        learner = ktrain.get_learner(model, train_data=trn, batch_size=120)
        learner.fit_onecycle(LR, EPOCHS)
        predictor = ktrain.get_predictor(learner.model, preproc)
        pred=predictor.predict(x_val)
        acc=accuracy_score(y_val,pred)
        print('acc',acc)
        accs.append(acc)

    y_pred=model.predict(x_val)
    print(classification_report(y_val.argmax(axis=-1), y_pred.argmax(axis=-1),target_names=possible_labels))
    return predictor, label_dict
    
    
def getMatchingResults(parent_class_instances, model, dictionary):
    # here we are applaying instances in the target KG to the trained model
    predected_class_list = []

    Instance_Names=parent_class_instances['name_t']
    predictions = model.predict(list(Instance_Names))
    p=np.argmax(predictions, axis=-1)
    for name in predictions:
        predected_class_list.append(name)

    parent_class_instances['Predected_Class']=predected_class_list

    return parent_class_instances


##############################################
#
#Place
#
##############################################

dataset=pd.read_csv('/data/lip18oaf/wdc_data/places_with_other_class.csv')


original=len(dataset)
dataset.drop_duplicates(subset=['schemaorg_class','name_t','page_domain'], keep='first', inplace=True, ignore_index=True)
new=len(dataset)
print('Total number of duplicate items is', original-new)


Model,label_dict=trainClassifiers(dataset)


##############################################
#
# Local Business
#
##############################################

dataset=pd.read_csv('/data/lip18oaf/wdc_data/localBusiness_with_other_class.csv')#localBusiness_


original=len(dataset)
dataset.drop_duplicates(subset=['schemaorg_class','name_t','page_domain'], keep='first', inplace=True, ignore_index=True)
new=len(dataset)
print('Total number of duplicate items is', original-new)


Model,label_dict=trainClassifiers(dataset)




##############################################
#
# Creative Work
#
##############################################


dataset=pd.read_csv('/data/lip18oaf/wdc_data/CreativeWork_with_other_class.csv')#localBusiness_


original=len(dataset)
dataset.drop_duplicates(subset=['schemaorg_class','name_t','page_domain'], keep='first', inplace=True, ignore_index=True)
new=len(dataset)
print('Total number of duplicate items is', original-new)


Model,label_dict=trainClassifiers(dataset)
