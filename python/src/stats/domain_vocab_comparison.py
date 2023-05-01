'''
This calculates two stats as described in the paper

intra-domain compression
cross-domain overlap
'''
import pandas as pd
import pickle
from exp import wdc_main_copy as wmc
from nltk import ngrams
from sandbox import domain_name_extractor as dne
import numpy

def count_class_per_domain(in_file, col_website, col_class):
    df = pd.read_csv(in_file, header=0, delimiter=',', quoting=0, encoding="utf-8")
    df = df[df.schemaorg_class!="other"]

    print(in_file)
    websites = sorted(list(df[col_website].unique()))
    print("\t Total websites={}".format(len(websites)))
    for w in websites:
        subset = df[(df[col_website] == w)]
        unique_classes = sorted(list(subset[col_class].unique()))
        if len(unique_classes)>1:
            print("\t\t{}={}".format(w,len(unique_classes)))

def intra_domain_compression(in_flle, col_text_data, col_class, col_website, out_folder, n, root_class):
    print(in_flle)
    df = pd.read_csv(in_flle, header=0, delimiter=',', quoting=0, encoding="utf-8")
    df = df[df.schemaorg_class!="other"]

    #go through each class
    unique_classes = sorted(list(df[col_class].unique()))
    data = {}
    for c in unique_classes:
        #go through each website
        print(c)
        subset = df[(df[col_class] == c)]
        websites = sorted(list(subset[col_website].unique()))

        scores=[]
        class_set=set()
        class_total=0
        for w in websites:
            sub_subset = subset[(subset[col_website] == w)]
            texts=prep_text_data(sub_subset, col_text_data)
            ngrams=extrac_ngrams(texts, n)
            score, website_set, website_doc_total=calculate_intra_domain_compression(ngrams)
            class_set.update(website_set)
            class_total+=website_doc_total
            print("\t\t{}={}".format(w, score))
            scores.append(score)
        data[c]=scores
        print("For class {}={}".format(c, (1-len(class_set)/class_total)))

    pickle.dump(data, open("{}/{}_{}_ngram={}.pk".format(out_folder, "stats_intra_domain_compression",root_class, n), "wb"))
    #create visualisation


def cross_domain_dice_overlap(in_flle, col_text_data, col_class, col_website, out_folder, n, root_class):
    print(in_flle)
    df = pd.read_csv(in_flle, header=0, delimiter=',', quoting=0, encoding="utf-8")
    df = df[df.schemaorg_class != "other"]

    # go through each class
    unique_classes = sorted(list(df[col_class].unique()))
    data = {}
    for c in unique_classes:
        # go through each website
        print(c)
        subset = df[(df[col_class] == c)]
        websites = sorted(list(subset[col_website].unique()))
        website_vocab={}
        dice_values={}
        for w in websites:
            sub_subset = subset[(subset[col_website] == w)]
            texts = prep_text_data(sub_subset, col_text_data)
            ngrams = extrac_ngrams(texts, n)
            website_vocab[w]= [item for sublist in ngrams for item in sublist]

        print("\t deriving website pairs")
        web_pairs=[]

        for i in range(0, len(websites)):
            for j in range(i+1, len(websites)):
                web_pairs.append((websites[i], websites[j]))
        print("\t total of {} pairs".format(len(web_pairs)))
        count=0
        for wbp in web_pairs:
            wb1 = set(website_vocab[wbp[0]])
            wb2 = set(website_vocab[wbp[1]])

            # web1 = dne.extract_domain_name_nosegment(wbp[0])
            # web1_segmented=dne.extract_domain_name(wbp[0])
            # web2 = dne.extract_domain_name_nosegment(wbp[1])
            # web2_segmented = dne.extract_domain_name(wbp[1])
            # dice_web_name = calculate_dice(set(web1_segmented), set(web2_segmented))
            # if dice_web_name>=0.8:
            #     print("attetion")

            dice =calculate_dice(wb1, wb2)
            #if dice>0:
                #data[wbp] =dice
            dice_values[wbp]=dice
            count+=1
            if count%5000==0:
                print("\t\t{}/{}".format(count, len(web_pairs)))
        dice_values= {k: v for k, v in sorted(dice_values.items(), key=lambda item: item[1], reverse=True)}
        dice_values_sorted=list(dice_values.values())
        print("\t for {}, max={} {}, min={}".format(c, dice_values_sorted[0],
                                                      list(dice_values.keys())[0],
                                                      dice_values_sorted[len(dice_values_sorted)-1]))
        data[c] = dice_values
        #print(dice_values)

    ##saving the data file
    pickle.dump(data, open("{}/{}_{}_ngram={}.pk".format(out_folder, "stats_cross_domain_dice",n, root_class), "wb"))
    # create visualisation
    #print("Now creating visualisation...")  # todo


def prep_text_data(dataframe, col_text_data):
    values=[]
    for index, row in dataframe.iterrows():
        text = row[col_text_data]
        text=wmc.clean_text(text).strip()
        if len(text)<2:
            continue
        values.append(text)
    return values

def extrac_ngrams(texts:list, n):
    output=[]
    for t in texts:
        grams=[]
        splits=t.split()
        if n==1:
            grams.extend(splits)
        if n>=2:
            bigrams = ngrams(splits, 2)
            grams.extend(unpack(bigrams))
        if n>=3:
            trigrams=ngrams(splits, 3)
            grams.extend(unpack(trigrams))
        if n>=4:
            fourgrams=ngrams(splits,4)
            grams.extend(unpack(fourgrams))
        output.append(grams)

    return output

def unpack(grams):
    res=[]
    for g in grams:
        res.append(" ".join(g))
    return res

def calculate_dice(set1, set2):
    set3 = set1.intersection(set2)
    return 2* len(set3)/ (len(set1)+len(set2))

def calculate_intra_domain_compression(ngrams):
    entire_set=set()
    total=0
    for entry in ngrams:
        doc_set=set(entry)
        entire_set.update(doc_set)
        total+=len(doc_set)
    return 1- len(entire_set)/total, entire_set, total



if __name__ == "__main__":
    #count_class_per_domain
    #count_class_per_domain("/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/CreativeWork.csv",
    #                       "page_domain","schemaorg_class")
    #count_class_per_domain("/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/LocalBusiness.csv",
    #                       "page_domain", "schemaorg_class")
    #count_class_per_domain("/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/Place.csv",
    #                       "page_domain", "schemaorg_class")

    #n=1
    #intra_domain_compression("/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/CreativeWork.csv",
    #                       "description", "schemaorg_class","page_domain",
    #                         "/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/domain_stats",n, "creativework")
    #intra_domain_compression("/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/LocalBusiness.csv",
    #                         "description", "schemaorg_class", "page_domain",
    #                         "/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/domain_stats",n,"localbusiness")
    #intra_domain_compression("/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/Place.csv",
    #                         "description", "schemaorg_class", "page_domain",
    #                         "/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/domain_stats",n, "place")

    for n in range(1, 5):
        cross_domain_dice_overlap("/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/CreativeWork.csv",
                             "description", "schemaorg_class", "page_domain",
                             "/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/domain_stats", n,"creativework")
        cross_domain_dice_overlap("/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/LocalBusiness.csv",
                             "description", "schemaorg_class", "page_domain",
                             "/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/domain_stats", n, "localbusiness")
        cross_domain_dice_overlap("/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/Place.csv",
                             "description", "schemaorg_class", "page_domain",
                             "/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/domain_stats", n,"place")