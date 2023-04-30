'''
takes the n-fold dataset created by split_by_doman.py, normalise domain names and save only the data needed
'''
import pandas as pd
from pathlib import Path
from sandbox import domain_name_extractor as dne

def convert(inFolder, col_label, col_website, col_name, col_1sent):
    for path in Path(inFolder).rglob('*.csv'):
        p = str(path)
        print(p)
        df = pd.read_csv(p, header=0, delimiter=',', quoting=0, encoding="utf-8")

        dfnew = df[[col_label, col_name, col_1sent, col_website]]
        for i, row in dfnew.iterrows():
            domain = row[col_website]
            domain=dne.extract_domain_name(domain)
            dfnew.at[i, col_website] = row[col_name]+" "+domain
        dfnew.to_csv(p, sep=',', encoding='utf-8')



if __name__ == "__main__":
    convert("/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/website_folds",
            "schemaorg_class", "page_domain", "name_t", "description"
            )
