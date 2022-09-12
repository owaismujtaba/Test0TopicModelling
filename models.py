import gensim
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim import corpora
from utils import idtoword
import os
from gensim.models import CoherenceModel
from bertopic import BERTopic

def LDA(documents, n_topics=5):
    
    """
    Fit and LDA model on the data
    Args:
        documents: A 2d list of cleaned documents containing tokens in each document
        n_topics: number of topics or clusters
    
    """
    print("Fitting the LDA Model")
    id2word = corpora.Dictionary(documents)
    corpus = idtoword(id2word, documents)
    
    model = gensim.models.ldamodel.LdaModel(
        corpus = corpus,
        id2word=id2word,
        num_topics=n_topics,
        random_state=100,
        update_every =1,
        chunksize=100,
        passes=10,
        alpha='auto')
    
    vis = gensimvis.prepare(
            model, 
            corpus, 
            id2word, 
            mds="PCoA",
            R=30)
    
    
    pyLDAvis.save_html(vis, os.getcwd()+'/Results/'+str(n_topics)+'lda.html')
    print("Results saved in Results Folder")
    
    model_path = os.getcwd()+'/Models/'+str(n_topics)+'LDAmodel.model'
    
    model.save(model_path)
    
    
    coherence_model_lda = CoherenceModel(
    model=model, texts=corpus, dictionary=id2word, coherence='c_v')

    coherence_lda = coherence_model_lda.get_coherence()

    print('\nCoherence Score: ', coherence_lda)

    print("Model Saved: ", model_path)
    #cluster_test(corpus, model)
    
    
    
def bert_model(texts):
    
    model = BERTopic()
    model = BERTopic(embedding_model="all-MiniLM-L6-v2")
    model, probs = model.fit_transform(texts)
    
    model_path = os.getcwd()+'/Models/'+str(n_topics)+'BERTmodel.model'
    model.save(model_path)
    
    