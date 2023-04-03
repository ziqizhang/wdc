'''
Analyse
- for each schemaorg label/class, and each class-website pair:
-- name token #, description token #, first sentence token #, website token #
'''
import numpy, json, pandas, sys,re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))


def calc_stats(df, col_name, col_description, col_firstsent, col_website, col_label, parent_class, outfolder):
    #work out unique labels and websites
    unique_labels=set(df[col_label].unique())
    unique_labels.remove("other")
    label_to_websites={}

    for l in unique_labels:
        label_to_websites[l]=df.loc[df[col_label] == l][col_website].unique()

    label_to_websites=sorted(label_to_websites.items())

    json_data_name=[]
    json_data_desc=[]
    json_data_firstsent=[]

    #query each label and website and output data
    for k, v in label_to_websites:
        subset=df[df[col_label]==k]
        name_stats, desc_stats, first_sent_stats=\
            calc_name_desc_firstsent_stats(subset, k, col_name, col_description, col_firstsent,"")
        json_data_name.append(name_stats)
        json_data_desc.append(desc_stats)
        json_data_firstsent.append(first_sent_stats)

        websites = sorted(list(v))

        json_data_name_per_schemaorglabel=[]
        json_data_desc_per_schemaorglabel = []
        json_data_firstsent_per_schemaorglabel = []
        for w in websites:
            subset=df[(df[col_label]==k) & (df[col_website]==w)]
            name_stats, desc_stats, first_sent_stats = \
                calc_name_desc_firstsent_stats(subset, w, col_name, col_description, col_firstsent,"\t")
            json_data_name_per_schemaorglabel.append(name_stats)
            json_data_desc_per_schemaorglabel.append(desc_stats)
            json_data_firstsent_per_schemaorglabel.append(first_sent_stats)
        #outupt for website data per schemaorg class
        save_json(json_data_name_per_schemaorglabel, '{}/{}_by_website_name.json'.format(outfolder, k))
        save_json(json_data_desc_per_schemaorglabel, '{}/{}_by_website_desc.json'.format(outfolder, k))
        save_json(json_data_firstsent_per_schemaorglabel, '{}/{}_by_website_1stsent.json'.format(outfolder, k))

    #ouput per schemaorg class data
    save_json(json_data_name, '{}/{}_overall_name.json'.format(outfolder, parent_class))
    save_json(json_data_desc, '{}/{}_overall_desc.json'.format(outfolder, parent_class))
    save_json(json_data_firstsent, '{}/{}_overall_1stsent.json'.format(outfolder, parent_class))

def calc_name_desc_firstsent_stats(subset, schemaorg_label, col_name, col_description, col_firstsent,indent):
    print("{}Processing {}".format(indent, schemaorg_label))
    name_toks = []
    desc_toks = []
    first_sent_toks = []
    for index, row in subset.iterrows():
        name = row[col_name]
        desc = row[col_description]
        firstsent = row[col_firstsent]
        if firstsent.startswith(name):
            firstsent = firstsent[len(name):].strip()

        clean = clean_text(name)
        name_toks.append(len(re.split(r"\s+", clean)))
        clean = clean_text(desc)
        desc_toks.append(len(re.split(r"\s+", clean)))
        clean = clean_text(firstsent)
        first_sent_toks.append(len(re.split(r"\s+", clean)))

    name_stats = produce_boxplot_stats(name_toks)
    name_stats["label"] = schemaorg_label
    desc_stats = produce_boxplot_stats(desc_toks)
    desc_stats["label"] = schemaorg_label
    first_sent_stats = produce_boxplot_stats(first_sent_toks)
    first_sent_stats["label"] = schemaorg_label

    return name_stats, desc_stats, first_sent_stats

def save_json(data, file):
    str_ = json.dumps(data,
                      indent=4, default=str)
    with open(file,'w') as outfile:
        outfile.write(str_)

def clean_text(text):
    """
    keep this method same as that in wdc_main
        text: a string

        return: modified initial string
    """
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))

    text = BeautifulSoup(text, "lxml").text  # HTML decoding
    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub(' ', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # delete stopwors from text
    text = ' '.join([i for i in text.split() if not i.isdigit()])
    new = text.replace("_", " ")
    return new

def produce_boxplot_stats(values):
    return {"mean": numpy.mean(values),
     "med": numpy.median(values),
     "q1": numpy.quantile(values, .25),
     "q3": numpy.quantile(values, .75),
     "whislo": numpy.min(values),
     "whishi": numpy.max(values)}

if __name__ == "__main__":

    #input file should point to raw input like place.csv, localbusiness.csv...
    #ol_name, col_description, col_firstsent, col_website, col_label, col_verification, outfolder
    df = pandas.read_csv(sys.argv[1], header=0, delimiter=',', quoting=0, encoding="utf-8",dtype=str)
    calc_stats(df,
               "name_t",
               "description_t",
               "description",
               "page_domain",
               "schemaorg_class",
               sys.argv[3],
               sys.argv[2])