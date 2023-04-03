from sklearn.metrics import cohen_kappa_score
import pandas as pd

in_file="/home/zz/Data/wdc_data_index/wdctable_202012_index_top100_export_forML/IAA_sample_shared.csv"
df = pd.read_csv(in_file, header=0, delimiter=',', quoting=0, encoding="utf-8")
array1 = list(df.iloc[:, 5])
array2 = list(df.iloc[:,6])

print(cohen_kappa_score(array1, array2))