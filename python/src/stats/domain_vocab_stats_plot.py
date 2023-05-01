'''
Takes output produced by domain_vocab_comparison and produces states in json for visualisation
'''
import os, pickle, numpy
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sandbox import domain_name_extractor as dne

def produce_boxplot_stats(values):
    return {"mean": numpy.mean(values),
     "med": numpy.median(values),
     "q1": numpy.quantile(values, .25),
     "q3": numpy.quantile(values, .75),
     "whislo": numpy.min(values),
     "whishi": numpy.max(values),
            "fliers":[]}

'''
Expecting pickle files, see output by 'intra_domain_compression' from domain_vocab_comparison
dictionary with 
- key = class
- value = a list of values
'''
def intra_domain_stats_plot(infolder):
    labelsize = 10
    rcParams['xtick.labelsize'] = labelsize
    rcParams['ytick.labelsize'] = labelsize

    for f in os.listdir(infolder):
        if not f.endswith("pk"):
            continue

        data = pickle.load(open(infolder+"/"+f, "rb"))
        sorted_keys = sorted(list(data.keys()))
        stats=[]
        for k in sorted_keys:
            item = produce_boxplot_stats(data[k])
            item["label"]=k
            stats.append(item)

        fs = 12  # fontsize

        fig, axes = plt.subplots(nrows=1, ncols=1, sharey=True)
        axes.bxp(stats)
        axes.set_title('Boxplot for {}'.format(f), fontsize=fs)
        plt.xticks(rotation=90)
        plt.tight_layout()
        # plt.subplots_adjust(bottom=0.3)
        plt.savefig(infolder + "/" + f + ".png", format='png', dpi=300)
        plt.clf()
        plt.close(fig)

'''
Takes output from cross_domain_dice_overlap
A dictionary with
- key = class
- value = dictionary with pair of websites, and dice score

'''
def cross_domain_stats_plot(infolder):
    #filter pairs, if they are the same host, ignore
    labelsize = 10
    rcParams['xtick.labelsize'] = labelsize
    rcParams['ytick.labelsize'] = labelsize

    for f in os.listdir(infolder):
        if not f.endswith("pk"):
            continue
        data = pickle.load(open(infolder + "/" + f, "rb"))
        sorted_keys = sorted(list(data.keys()))
        stats = []


        for k in sorted_keys:
            #produce stats for the dictionary
            dice_scores = []
            for pair, dice in data[k].items():
                if dice < 0.5:
                    dice_scores.append(dice)
                else:
                    w1 = pair[0]
                    w2 = pair[1]
                    web1 = dne.extract_domain_name_nosegment(w1)
                    web1_segmented = dne.extract_domain_name(w1)
                    web2 = dne.extract_domain_name_nosegment(w2)
                    web2_segmented = dne.extract_domain_name(w2)
                    dice_web_name = calculate_dice(set(web1_segmented), set(web2_segmented))
                    if dice_web_name < 0.8:
                        dice_scores.append(dice)
                    else:
                        print(pair)
            item=produce_boxplot_stats(dice_scores)
            item["label"] = k
            stats.append(item)

        fs = 12  # fontsize

        fig, axes = plt.subplots(nrows=1, ncols=1, sharey=True)
        axes.bxp(stats)
        axes.set_title('Boxplot for {}'.format(f), fontsize=fs)
        plt.xticks(rotation=90)
        plt.tight_layout()
        # plt.subplots_adjust(bottom=0.3)
        plt.savefig(infolder + "/" + f + ".png", format='png', dpi=300)
        plt.clf()
        plt.close(fig)

def calculate_dice(set1, set2):
    set3 = set1.intersection(set2)
    return 2* len(set3)/ (len(set1)+len(set2))

if __name__ == "__main__":
    #intra_domain_stats_plot("/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/domain_stats/intra_domain")
    cross_domain_stats_plot(
        "/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/domain_stats/cross_domain")