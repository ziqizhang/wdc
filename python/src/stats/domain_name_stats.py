import pandas as pd
from sklearn.model_selection import StratifiedKFold
def UnknownHosts(dataset):
    dataset = dataset.drop(dataset.index[dataset.name_t.str.contains(r'[0-9]', na=False)])
    dataset = dataset[dataset['name_t'].notnull()]

    # dataset['description'] = dataset['description'].apply(clean_text)

    # parse the domain name
    # dataset['page_domain']=dataset['page_domain'].apply(extract_domain_name)

    # add the name + domain column
    dataset['domain1'] = dataset['name_t'] + ' ' + dataset['page_domain']

    # add the name + 1 scentence _ domain column
    dataset['domain2'] = dataset['description'] + ' ' + dataset['page_domain']

    # this is the column to be used to train the model
    # 'name_t' for name only
    # 'description' for name + 1 scenternce of description
    # 'domain1' for name and page domain (parsed)
    # 'domain2' for name  1 scenternce of description and page domain (parsed)

    Training_Column = 'description'

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
    Hosts_stats=pd.DataFrame(columns=['class_name','unique_hosts'])

    for class_name in possible_labels:
        for train_index, val_index in Folds.split(data, data['schemaorg_class']):
            train, val = dataset.iloc[train_index], dataset.iloc[val_index]
            train_unique_hosts=set(pd.unique(train['schemaorg_class'==class_name]['page_domain']))
            test_unique_hosts=set(pd.unique(val['schemaorg_class'==class_name]['page_domain']))

            unique_hosts=test_unique_hosts.difference(train_unique_hosts)

        Hosts_stats.append({'class_name':class_name, 'unique_hosts':unique_hosts},ignore_index='True')

    return Hosts_stats


if __name__ == "__main__":
    dataset=pd.read_csv('/Users/omaimaaf/Desktop/WDCdata/wdc_data_v4/Place.csv')
    #parent=pd.read_csv('/data/lip18oaf/wdc_data_v4/Place_parent.csv')

    original=len(dataset)
    dataset.drop_duplicates(subset=['schemaorg_class','name_t','page_domain'], keep='first', inplace=True, ignore_index=True)
    new=len(dataset)
    print('Total number of duplicate items is', original-new)
    uh=UnknownHosts(dataset)

    

