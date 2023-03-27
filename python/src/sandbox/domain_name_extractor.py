from urllib.parse import urlparse
import pandas as pd
import wordsegment as ws
import numpy as np

##if you adapt this code make sure to call this method in your new code before using the method below
ws.load()
stopwords=("www","ww","http","https","ftp","co","uk","com","net","org","nl","gov","in","ca","ie","nz","au","za",
           "qa","th","bh","sa","a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r",
           "s","t","u","v","w","x","y","z","sg","ph")

def extract_domain_name(str):
    try:
        if(len(str))==0 or str is np.nan:
            return ""
        str="http://"+str
        obj=urlparse(str)
        hostname=obj.hostname
        words=set(hostname.split("."))
        words=words.difference(stopwords)
        words=" ".join(words)
        words=set(ws.segment(words))
        words = words.difference(stopwords)

        return " ".join(words)

    except:
        print("Cannot parse {}".format(str))
        return ""

if __name__ == "__main__":
    col_domain=6
    in_file="/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/LocalBusiness.csv"
    df = pd.read_csv(in_file, header=0, delimiter=',', quoting=0, encoding="utf-8")
    domains=pd.unique(df['page_domain'])
    for d in domains:
        v = extract_domain_name(d)
        print("{} => {}".format(d, v))
