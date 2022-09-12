import os
import pandas as pd
import pdb
DATA_PATH = os.getcwd()+'/Data/data.csv'



def load_dataset(file_path= DATA_PATH):
    
    documents = pd.read_csv(file_path)
    #pdb.set_trace()
    texts = []
    for doc in documents['text']:
        texts.append(str(doc))
    
    return texts
    