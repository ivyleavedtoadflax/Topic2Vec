
# coding: utf-8

# # TOPIC2VEC algorithm by using gensim and according to the second hint given by Gordon Mohr.  
# (https://groups.google.com/forum/#!topic/gensim/BVu5-pD6910)
# 
# 
# 1. Vectorization of docs by using CountVectorizer (with or without tfidf) with no lemmatization
# 2. Latent Dirichlet Allocation 
# 3. Topic2Vec of the entire dataset (20 NewsGroups)   
# 
# It saves:
# * the topic2vec model obtained

# Imports

# In[1]:


import numpy as np; import pandas as pd; import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import codecs 
from glob import glob
import os
import pickle
import copy
import pyorient
import ast


# In[25]:


from __future__ import print_function
import time
import string
import re
# random
from random import shuffle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation


# In[3]:


from gensim import corpora, models, similarities


# In[4]:


n_top_words = 20


# ## 1. IMPORTING DOCS FROM 20 NEWSGROUPS DATASET

# In[5]:


from sklearn.datasets import fetch_20newsgroups
categories = ['comp.sys.ibm.pc.hardware',
'comp.sys.mac.hardware',
'comp.windows.x',
'rec.sport.baseball',
'rec.sport.hockey',
'sci.med',
'sci.space',
'soc.religion.christian']

n_topics = len(categories)

categories_source = {}

for cat in categories:
    categories_source[cat] = cat.replace('.', '_')


# In[6]:


newsgroups_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers', 'quotes'),
                                      categories=categories)


# In[7]:


for i,j in categories_source.items():
    print(i,j)


# #### TOTAL NUMBER OF DOC

# In[8]:


n_docs = newsgroups_train.filenames.shape[0]
n_docs


# # 2. LDA to find the topic most-associated with each word

# ## 2.1 From Strings to Vectors

# ### WITH Lemmatization
# Return the casting of the original tag in a single
# character which is accepted by the lemmatizer
import nltk.corpus  # splits on punctuactions   
stop_words = nltk.corpus.stopwords.words('english')

import re
def get_wordnet_pos(treebank_tag):

    # I recognize the initial character of the word, identifying the type
    if treebank_tag.startswith('J'):
        return nltk.corpus.reader.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.reader.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.reader.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.reader.wordnet.ADV
    else:
        return None

from nltk import word_tokenize, pos_tag        
from nltk.stem import WordNetLemmatizer 
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        tokenized_doc = word_tokenize(doc) # splits on punctuactions  
        tagged_doc = pos_tag(tokenized_doc)
        
        lemmatized_doc = []
        # Scan the (word, tag) tuples which are the elements of tagged_tweet1
        for word, tag in tagged_doc:
            ret_value = get_wordnet_pos(tag)
            # If the function does not return None I provide the ret_value
            if ret_value != None:
                lemmatized_doc.append(self.wnl.lemmatize(word, get_wordnet_pos(tag)))
            # If the function returns None I do not provide the ret_value
            else:
                lemmatized_doc.append(self.wnl.lemmatize(word))
        nonumbers_nopunct_lemmatized_doc = [word for word in lemmatized_doc if re.search('[a-zA-Z]{2,}', word)]
#        nonumbers_nopunct_lemmatized_doc = [word for word in nopunct_lemmatized_doc if not re.search('\d+', word)]
        lemmatized_doc_stopw = [word for word in nonumbers_nopunct_lemmatized_doc if word not in stop_words]
        
        return lemmatized_doc_stopw #[self.wnl.lemmatize(t) for t in word_tokenize(doc)]t0 = time()
tf_vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), encoding='utf-8', analyzer='word',
                                stop_words=["'s", "fx"], ngram_range = (1,1), min_df = 2).fit(df_norm2.text)
print("fit vectorizer with lemmatization done in %0.3fs." % (time() - t0))
# ### WITHOUT Lemmatization

# In[9]:


t0 = time()
tf_vectorizer = CountVectorizer(encoding='utf-8', analyzer='word', stop_words='english',
                                ngram_range = (1,1), min_df = 2, token_pattern = '[a-zA-Z]{2,}').fit(newsgroups_train.data)
print("fit vectorizer without lemmatization done in %0.3fs." % (time() - t0))


# ### Vectorization

# In[10]:


n_features = len(tf_vectorizer.get_feature_names())


# In[11]:


newsgroups_train.data[0]


# In[12]:


tf_docs = tf_vectorizer.transform(newsgroups_train.data)


# ### WITH TFIDF

# tfidf_vectorizer = TfidfTransformer(sublinear_tf=False, use_idf = True).fit(tf_docs)
# tfidf_docs = tfidf_vectorizer.transform(tf_docs)

# ## 2.2 LDA implementation

# In[13]:


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


# In[14]:


print("Fitting LDA models with tf features, n_samples=%d and n_features=%d..."
      % (n_docs, n_features))
lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online', learning_offset=50.,
                                random_state=0)
t0 = time()
lda.fit(tf_docs)
print("done in %0.3fs." % (time() - t0))


# In[15]:


print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)


# In[16]:


per_topic_distr_LDA = lda.components_
per_topic_distr_LDA.shape
#per_topic_distr_LDA.sum(axis=1)


# # 3. TOPIC2VEC

# In[17]:


most_p_topic = np.argmax(per_topic_distr_LDA, axis=0)


# In[18]:


word_and_topic = zip(tf_feature_names, most_p_topic)

word2topic_dict = {word : 'topic_' + np.array_str(topic) for word, topic in word_and_topic}


# ## 3.1 Tokenization

# In[19]:


def tokenizer(document):
    
    text = "".join([ch for ch in document if ch not in string.punctuation])
    text_list = text.split()
    normalized_text = [x.lower() for x in text_list]
    # Define an empty list
    nostopwords_text = []
    # Scan the words
    for word in normalized_text:
        # Determine if the word is contained in the stop words list
        if word not in ENGLISH_STOP_WORDS:
            # If the word is not contained I append it
            nostopwords_text.append(word)
    tokenized_text = [word for word in nostopwords_text if re.search('[a-zA-Z]{2,}', word)]
            
    return tokenized_text


# In[20]:


def map_doc_to_topic(tokenized_text, prefix, doc_id_number, word2topic_dict):
    doc_to_topic_list = [prefix + '_' + str(doc_id_number)]
    for word in tokenized_text:
        if word in word2topic_dict.keys():
            doc_to_topic_list.append(word2topic_dict[word])
            
    return doc_to_topic_list

class LabeledLineSentence(object):
    def __init__(self, docs_list, word2topic_dict):
        self.labels_list = word2topic_dict
        self.docs_list = docs_list
    def __iter__(self):
        for idx, doc in enumerate(self.docs_list):
            words_doc=tokenizer(doc)
            tags_doc = map_doc_to_topic(words_doc, idx, word2topic_dict)
            yield models.doc2vec.LabeledSentence(words = words_doc,
                                                 tags = tags_doc)
            
    def sentences_perm(self):
        shuffle(models.doc2vec.LabeledSentence)
        return models.doc2vec.LabeledSentence
# In[21]:


class LabeledLineSentence_training(object):
    def __init__(self, sources, word2topic_dict):
        self.labels_list = word2topic_dict
        self.sources = sources
        flipped = {}
        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')
    
    def __iter__(self):
        for source, prefix in self.sources.items():
            print(source)
            newsgroups_train_cat = fetch_20newsgroups(subset='train',
                                                      remove=('headers', 'footers', 'quotes'),
                                                      categories=[source])
            for idx, doc in enumerate(newsgroups_train_cat.data):
                words_doc=tokenizer(doc)
                tags_doc = map_doc_to_topic(words_doc, prefix, idx, word2topic_dict)
                yield models.doc2vec.LabeledSentence(words = words_doc,
                                                     tags = tags_doc)
                
    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            print(source)
            newsgroups_train_cat = fetch_20newsgroups(subset='train',
                                                      remove=('headers', 'footers', 'quotes'),
                                                      categories=[source])
            for idx, doc in enumerate(newsgroups_train_cat.data):
                words_doc=tokenizer(doc)
                tags_doc = map_doc_to_topic(words_doc, prefix, idx, word2topic_dict)
                self.sentences.append(models.doc2vec.LabeledSentence(words = words_doc,
                                                     tags = tags_doc))
        return self.sentences
            
    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


# ## 3.1 Training

# In[22]:


it = LabeledLineSentence_training(categories_source, word2topic_dict)


# In[23]:


model = models.Doc2Vec(size=100, window=10, min_count=4, dm=1, dbow_words=1,
                              workers=50, alpha=0.025, min_alpha=0.025) # use fixed learning rate
model.build_vocab(it.to_array())


# In[26]:


for epoch in range(10):
    start = time.time()
    
    model.train(it.sentences_perm(), total_examples=model.corpus_count, epochs=model.iter)
    model.alpha -= 0.002 # decrease the learning rate
    model.min_alpha = model.alpha # fix the learning rate, no decay
    
    stop = time.time()
    duration = stop-start
    print('epoch:', epoch, ' duration: ', duration)


# In[30]:


fname =  os.getcwd() # Prints the working directory
fname = fname + '/topic2vec_20NG_2_ndoc' + str(n_docs) + 'n_topic' + str(n_topics) + '.model'
model.save(fname)
print(fname)

