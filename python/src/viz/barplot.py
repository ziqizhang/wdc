from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
import os
#plt.rcParams.update({'font.size': 14})


labelsize = 10
rcParams['xtick.labelsize'] = labelsize
rcParams['ytick.labelsize'] = labelsize

def plot(in_folder,outfolder):

    for filename in os.listdir(in_folder):
        if not os.path.isdir(filename):
            if filename == '.DS_Store':
                continue
        df=pd.read_csv(in_folder+ '/'+filename)
        fig, axes = plt.subplots(nrows=1, ncols=1, sharey=True)
        stat = df.groupby('Predected_Class').agg({'name_t': 'count'})
        # plt.pie(stat.name_t, labels=stat.index)
        plt.bar(stat.index, stat.name_t)
        plt.xlabel("schema.org Class")
        plt.ylabel("Parent Class Instances")
        plt.xticks(rotation=90)
        plt.tight_layout()
        f=in_folder.rsplit('/',1)[1] + '_' + filename.rsplit('.')[0]
        plt.savefig(outfolder + "/" + f + ".png", format='png', dpi=300)
        plt.clf()
    #plt.show()

    plt.close(fig)


if __name__ == "__main__":

    in_folder = "/Users/omaimaaf/Desktop/WDCdata/Results/ParentClassResults/"
    out_folder = "/Users/omaimaaf/Desktop/WDCdata/Results/ParentClassPlots/"
    for filename in os.listdir(in_folder):
        if filename == '.DS_Store':
            continue
        plot(in_folder+filename, out_folder)