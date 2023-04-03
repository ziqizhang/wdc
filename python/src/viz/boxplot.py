import json, os

import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import sys, pickle


labelsize = 10
rcParams['xtick.labelsize'] = labelsize
rcParams['ytick.labelsize'] = labelsize

def plot(in_file,outfolder):
    f = open(in_file)
    json_data=json.load(f)
    f.close()
    start=in_file.rindex("/")
    end=in_file.rindex(".")
    f= in_file[start+1:end]

    stats=[]
    for item in json_data:
        item['mean']=float(item['mean'])
        item['med'] = float(item['med'])
        item['q1'] = float(item['q1'])
        item['q3'] = float(item['q3'])
        item['whislo'] = float(item['whislo'])
        item['whishi'] = float(item['whishi'])
        item['fliers']=[]
        stats.append(item)

    fs = 12  # fontsize

    fig, axes = plt.subplots(nrows=1, ncols=1, sharey=True)
    axes.bxp(stats)
    axes.set_title('Boxplot for {}'.format(f), fontsize=fs)
    plt.xticks(rotation=90)
    plt.tight_layout()
    #plt.subplots_adjust(bottom=0.3)
    plt.savefig(outfolder+"/"+f+".png", format='png', dpi=300)
    plt.clf()
    plt.close(fig)

def plot_from_pickle(in_file,outfile):
    f = open(in_file, 'rb')
    data = pickle.load(f)
    f.close()

    columns = [data['fake'], data['real']]
    fig, ax = plt.subplots()
    ax.boxplot(columns)
    plt.xticks([1, 2], ["Fake", "Real"])
    plt.savefig(outfile, format='png', dpi=300)
    #plt.show()


def plot_from_pickle2(in_file,outfile, minfake=0):
    f = open(in_file, 'rb')
    data = pickle.load(f)
    f.close()

    days=0
    data_fake=[]
    data_real=[]
    for k, v in data.items():
        data_fake.append(v['fake'])
        if v['fake']>minfake:
            days+=1
        data_real.append(v['real'])

    columns = [data_fake, data_real]
    fig, ax = plt.subplots()
    ax.boxplot(columns)
    plt.xticks([1, 2], ["#Fake", "#Real"])
    plt.savefig(outfile, format='png', dpi=300)
    print(days)
    #plt.show()


if __name__ == "__main__":
    # in_folder="/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/distribution/place"
    # out_folder="/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/distribution/place_plots"

    #in_folder = "/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/distribution/creativework"
    #out_folder = "/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/distribution/creativework_plots"
    #
    in_folder = "/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/distribution/localbusiness"
    out_folder = "/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/distribution/localbusiness_plots"
    for filename in os.listdir(in_folder):
        f=in_folder+"/"+filename
        plot(f, out_folder)

    # infile="/home/zz/Work/data/amazon/labelled/stats/stats_descriptive_rating.json"
    # outfolder="/home/zz/Work/data/amazon/labelled/stats/boxplots_rating"
    # plot(infile, outfolder)

    # infile = "/home/zz/Work/data/amazon/labelled/stats/stats_descriptive_words.json"
    # outfolder = "/home/zz/Work/data/amazon/labelled/stats/boxplots_length"
    # plot(infile, outfolder)

