import spacy
from nltk.corpus import stopwords
import string
from gensim import corpora
import gensim
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import pdb
import huspacy
import os

#huspacy.download()
def hungarian_stopwords():
    '''
    
    
    '''
    stops = []
    with open(os.getcwd()+'/stopshungarian.txt', 'r') as file:
        for line in file:
            line = line.replace("\n", "")
            stops.append(line)

    return stops



def clean_documents(documents):
    """
    Clean the raw text in the documents and retain only the necessary tokens
    Args:
        documents: A 2d list of textual documents
        
    Return:
        cleaned_docs: A 2d list of cleaned documents
    """
    
    
    print("Cleaning documents ")
    nlp = spacy.load('hu_core_news_lg')
    stops = hungarian_stopwords()
    
    
    cleaned_docs = []
   
    for doc in documents:
        doc = nlp(doc)
        tokens =[]
        
        for token in doc:
            #print(token, token.tag_)
            # Keeping only nouns verbs adjectives and whose length is greater than 2
            if token.tag_ in ['NNP', 'VBZ', 'VBG', 'IN', 'ADJ', 'PROP'] and str(token.text).lower() not in stops and len(token.text)>2:
                tokens.append(token.lemma_.lower())
        
        cleaned_docs.append(tokens)
    #pdb.set_trace()
    #data = {'text', clean}
    print("Documents cleaned")   
    return cleaned_docs




def idtoword(id2word, documents):
    
    """
    Gives the corpus from the documents
    Args:
        id2word: corpora Dictionary fitted on documents
        documents: A 2d list of cleaned documents containing tokens in each document
        
    Return:
        corpus: Corpus of words
    """
    
    corpus = []
    
    for text in documents:
        new = id2word.doc2bow(text)
        corpus.append(new)
    return corpus
    
    
def make_bigrams_trigrams(documents):
    
    #pdb.set_trace()
    bigram_phrases = gensim.models.Phrases(documents, min_count=5, threshold=50)
    trigram_phrases = gensim.models.Phrases(bigram_phrases[documents], min_count=5, threshold=50)
    
    bigram = gensim.models.phrases.Phraser(bigram_phrases)
    trigram = gensim.models.phrases.Phraser(trigram_phrases)
    
    bigram_trigram_documents = []
    for doc in documents:
        
        doc_bigrams = bigram[doc]
        doc_bigrams_trigrams = trigram[doc]
        bigram_trigram_documents.append(doc_bigrams_trigrams)
        
    
    return documents
        