'''
code to take data prepared by omaima: https://drive.google.com/drive/u/1/folders/1yI_tnvRW3ko7RNry4-yn2D8Mt6734Hjq
and split them to create 5 folds following the rules:

- delete 'other' class
- delete classes where there is only one domain host
- for each class, create the splits where there are no overlap of domains between them
- create five folds and output
'''

import pandas as pd
import random, os, collections

random.seed(42)

def split(inFile, col_label, col_website, outfolder, nfold=5):
    df= pd.read_csv(inFile, header=0, delimiter=',', quoting=0, encoding="utf-8")
    unique_labels = set(df[col_label].unique())
    unique_labels.remove("other")

    #for each label, get unique set of websites
    label_to_websites={}
    for l in unique_labels:
        subset = df[(df[col_label] == l)]
        websites=set(subset[col_website].unique())
        label_to_websites[l]=websites

    #print stats of every class and website count
    for cls, splits in label_to_websites.items():
        print("{} = {}".format(cls, len(splits)))

    #create splits for each class based on domain
    label_to_splits={}
    for cls, splits in label_to_websites.items():
        websites=list(splits)
        if len(websites)>=nfold: #if a class has > fold number
            random.Random(42).shuffle(websites)
            splits = list(split_list(websites, nfold))
            label_to_splits[cls]=splits
        else:
            #we cant create n splits, create as many as possible
            splits=[]
            for w in websites:
                splits.append([w])
            label_to_splits[cls]=splits

    label_to_splits = collections.OrderedDict(sorted(label_to_splits.items()))

    #create folds and output
    for i in range(0, nfold):
        nextfolder=outfolder+"/"+str(i)
        print("Folder {}, saving into {}".format(i, nextfolder))
        os.makedirs(nextfolder, exist_ok=True)
        df_fold_train=[]
        df_fold_test=[]

        for cls, splits in label_to_splits.items():
            #work out the test set ids
            if i>=len(splits): #this class has less than nfold websites so there are not that many folds
                test_split_index=0 #we just assign an arbitrary split index
            else:
                test_split_index=i

            test_set_df=select_class_and_website(df, cls, splits[test_split_index], col_label, col_website)
            df_fold_test.append(test_set_df)
            train_split_indeces= []
            for x in range(0, nfold):
                if x!=test_split_index and x < len(splits):
                    train_split_indeces.append(x)
            train_set_websites=[]
            for j in train_split_indeces:
                train_set_websites.extend(splits[j])
            train_set_df=select_class_and_website(df, cls, train_set_websites, col_label, col_website)
            df_fold_train.append(train_set_df)

            # check overlap of websites again
            inter = set(splits[test_split_index]).intersection(set(train_set_websites))
            if len(inter) > 0:
                print("\t\t\toverlap detected in website lists: {}".format(inter))
            # check overlap of df again
            unique_websites_train = set(train_set_df[col_website].unique())
            unique_websites_test = set(test_set_df[col_website].unique())
            inter = unique_websites_test.intersection(unique_websites_train)
            if len(inter) > 0:
                print("\t\t\toverlap detected in df: {}".format(inter))

        df_test=pd.concat(df_fold_test)
        df_train = pd.concat(df_fold_train)
        #output
        total=len(df_train)+len(df_test)
        df_test.to_csv(nextfolder + "/test.csv", sep=',', encoding='utf-8')
        df_train.to_csv(nextfolder + "/train.csv", sep=',', encoding='utf-8')
        print("\t train:test ratio = {}:{}, total={}".format(len(df_train)/total, len(df_test)/total,total))

        #check train test website overlap
        check_class_website_pair_overlap(df_train, df_test, col_label, col_website)

def check_class_website_pair_overlap(df_train, df_test, col_class, col_website):
    train_pairs=set()
    test_pairs=set()
    for index, row in df_train.iterrows():
        train_pairs.add("{},{}".format(row[col_class], row[col_website]))

    for index, row in df_test.iterrows():
        test_pairs.add("{},{}".format(row[col_class], row[col_website]))

    inter=train_pairs.intersection(test_pairs)
    if len(inter)>0:
        print("\t\tOverlap detected: {}".format(inter))

    unique_labels_train=set(df_train[col_class].unique())
    unique_labels_test = set(df_test[col_class].unique())
    print("\t\tTrain has {} labels, test has {} labels".format(len(unique_labels_train), len(unique_labels_test)))


def check_intersection(list_of_splits):
    overlap=False
    for i in range(0, len(list_of_splits)):
        for j in range(i+1, len(list_of_splits)):
            set1 = set(list_of_splits[i])
            set2 = set(list_of_splits[j])
            inter = set1.intersection(set2)
            if len(inter)>0:
                overlap=True
                print("\t\tsplit {} and split {} shares websites:{}".format(i, j, inter))
    return overlap

def select_class_and_website(source_df, cls, websites:list, col_label, col_website):
    subset = source_df[(source_df[col_label] == cls) & (source_df[col_website].isin(websites))]
    return subset

def split_list(l, parts):
  #for i in range(0, len(list_a), chunk_size):
  #  yield list_a[i:i + chunk_size]
  n = min(parts, max(len(l), 1))
  k, m = divmod(len(l), n)
  return [l[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

if __name__ == "__main__":
    outfolder="/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/website_folds/Place"
    split("/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/Place.csv",
           "schemaorg_class","page_domain", outfolder, 5)
    print()
    outfolder = "/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/website_folds/LocalBusiness"
    split("/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/LocalBusiness.csv",
          "schemaorg_class", "page_domain", outfolder, 5)
    print()
    outfolder = "/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/website_folds/CreativeWork"
    split("/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/CreativeWork.csv",
          "schemaorg_class", "page_domain", outfolder, 5)

