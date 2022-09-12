from dataset import load_dataset
from utils import clean_documents
from utils import make_bigrams_trigrams
from models import LDA, bert_model


import pdb

def train_LDA():
    #pdb.set_trace()
    documents = load_dataset()
    
    cleaned_documents = clean_documents(documents)
    cleaned_documents = make_bigrams_trigrams(cleaned_documents)
    for topics in range(6, 10):
        print('Number of topics: ', topics)
        LDA(cleaned_documents, n_topics=topics)

        
        
def bert_train():
    
    documents = load_dataset()
    bert_model(documents)

    