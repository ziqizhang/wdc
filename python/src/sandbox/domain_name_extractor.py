from urllib.parse import urlparse
import pandas as pd
import wordsegment as ws

##if you adapt this code make sure to call this method in your new code before using the method below
ws.load()

def extract_domain_name(str):
    try:
        if(len(str))==0:
            return 0
        str="http://"+str
        obj=urlparse(str)
        hostname=obj.hostname
        if hostname.startswith("www."):
            hostname=hostname[4:]
        if "." in hostname:
            hostname=hostname[:hostname.index(".")]
        if len(hostname)>0:
            words=ws.segment(hostname)
        return hostname+" "+ " ".join(words)

    except:
        print("Cannot parse {}".format(str))
        return ""

if __name__ == "__main__":
    col_domain=6
    in_file="/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/Place.csv"
    df = pd.read_csv(in_file, header=0, delimiter=',', quoting=0, encoding="utf-8")
    domains=pd.unique(df['page_domain'])
    for d in domains:
        v = extract_domain_name(d)
        print("{} => {}".format(d, v))
