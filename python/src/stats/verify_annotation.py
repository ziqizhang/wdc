'''
Used to analyse the annotations verified by annotators
calculate for each class, % of correct ones, # of websites, # of total, % of correct ones per website
'''
import pandas as pd
import csv
def load_and_merge_data(*in_files):
    dataframes = []
    for in_file in in_files:
        dataframes.append(pd.read_csv(in_file, header=0, delimiter=',', quoting=0, encoding="utf-8",dtype=str))

    df = pd.concat(dataframes)
    return df

def calc_stats(df, col_label, col_website, col_verification, outfolder):
    #work out unique labels and websites
    unique_labels=df[col_label].unique()
    label_to_websites={}

    for l in unique_labels:
        label_to_websites[l]=df.loc[df[col_label] == l][col_website].unique()

    label_to_websites=sorted(label_to_websites.items())


    worksheet1_rows=[]
    worksheet2_rows=[]
    #query each label and website and output data
    for k, v in label_to_websites:
        total_with_label=len(df[df[col_label]==k])
        total_correct_with_label=len(df[(df[col_label]==k) & (df[col_verification]==1)])
        if total_correct_with_label==0:
            total_correct_with_label = len(df[(df[col_label] == k) & (df[col_verification] == "1")])

        worksheet1_rows.append([k, total_with_label, total_correct_with_label, total_correct_with_label/total_with_label])

        websites = sorted(list(v))
        for w in websites:
            total_with_label_from_website=len(df[(df[col_label]==k) & (df[col_website]==w)])
            total_correct_with_label_from_website=len(df[(df[col_label]==k) & (df[col_website]==w) & (df[col_verification]==1)])
            if total_correct_with_label_from_website==0:
                total_correct_with_label_from_website = len(
                    df[(df[col_label] == k) & (df[col_website] == w) & (df[col_verification] == "1")])

            worksheet2_rows.append(
                [k, w, total_with_label_from_website, total_correct_with_label_from_website,
                 total_correct_with_label_from_website / total_with_label_from_website])

    with open('{}/per_label.csv'.format(outfolder), 'w', newline='\n', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["Schemaorg_class","Total","Total correct","%"])
        for r in worksheet1_rows:
            writer.writerow(r)

    with open('{}/per_label_website_pair.csv'.format(outfolder), 'w', newline='\n', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["Schemaorg_class","Website","Total","Total correct","%"])
        for r in worksheet2_rows:
            writer.writerow(r)

if __name__ == "__main__":
    df=load_and_merge_data("/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/sample_1_annotated.csv",
                           "/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/sample_2_annotated.csv")
    calc_stats(df,"schemaorg_class","page_domain","Label",
               "/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML")