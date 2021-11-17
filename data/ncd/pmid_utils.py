
#%%
import pandas as pd
import numpy as np


""" Test file to generate sentences from file """
excelData = pd.read_excel('Literatura_biomedica_PubMed.xlsx', index_col=None, header=0)


""" Now split sentences  """
diseases_list = list()
diseases_pmids = list()
disease_names = list()
diseases_df = pd.DataFrame()
for idx ,item in excelData.iterrows():
    print(idx)
    disease_name = item['Term'].strip()
    abstract  = item['Abstract']  # for the nCD
    pmid = item['PMID'] # for ncd
    diseases_list.append(disease_name)
    diseases_pmids.append(pmid)
    if disease_name not in disease_names:
        disease_names.append(disease_name)

pmid_labels = dict((zip(diseases_pmids,diseases_list)))
# pd.DataFrame(pmid_labels).to_csv('pmid_labels.csv',header=None,index=None)
print(disease_names)
np.save('pmid_labels.npy', pmid_labels) 
#%%
