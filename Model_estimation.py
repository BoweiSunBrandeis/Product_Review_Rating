
# coding: utf-8

# In[1]:


# required model inport
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from nltk.corpus import movie_reviews
import nltk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import brown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from pickle import dump
from pickle import load
from sklearn import metrics
from load_data import load_data
import matplotlib.pyplot as plt


# In[4]:


# import data
data_labled = load_data(1000)


# In[5]:


# construct the word set that used to features extraction.

# all words in nltk corpus
all_words = list(set(nltk.corpus.words.words()))

# words with content meaning
content_words = contentextract(all_words)


# In[3]:


# negation function 
def negation(document):
    """This function deal with the negation in the reviews."""
    negation=[]
    for item in document:
        if '\'t' in item:
            negation.append('not')
            negation.append(item)
        else:
            negation.append(item)
    return negation


# In[4]:


# extract words with meaningful content
def contentextract(text):
    """This funtion get rid of the non-sense word in the reviews."""
    stopword=stopwords.words('english')
    punctuation=['.','?','!',',',';',"''",'""',"'","--","``","|","<",">","...","......","'s",':','[',']',
                 '(',')','#','*','$','%','@','^','-','+','=','/','{','}','\\','"','&']
    content=[w for w in text if w.lower() not in stopword]
    content=[w for w in content if w not in punctuation]
    return content


# In[6]:


def feature_label(features):
    """Separate the features and labels"""
    f=[]
    l=[]
    for item in features:
        f.append(item[0])
        l.append(item[1])
    return f,l


# In[7]:


def features_extract(document, wordset):
    """Extract the raw count of tokens in a given document. The feature sequence is based on the
    sequence of all words"""
    words_doc = nltk.FreqDist(document)
    features = []
    for word in wordset:
        features.append(words_doc[word])
    return features


# In[8]:


# Bayes Model training 
def bayes_model(feature_train, help_rank_train, model_name):
    """Construct a bayes model."""
    model = MultinomialNB()
    model.fit(feature_train, help_rank_train)
    modelpkl = open(model_name,'wb')
    dump(model, modelpkl, -1)
    return


# In[9]:


# Tree Classifier Model training
def tree_model(feature_train, help_rank_train, model_name):
    """Construct a tree classifier model."""
    decision_tree = DecisionTreeClassifier()
    decision_tree = decision_tree.fit(feature_train, help_rank_train)
    tree_model = open(model_name,'wb')
    dump(decision_tree, tree_model, -1)
    return


# In[10]:


# cross validation
def cross_validation(feature_train, help_rank_train, model_name):
    """Construct a model through cross validation"""
    clf = svm.SVC(kernel='linear', C=1).fit(feature_train, help_rank_train)
    clf_model = open(model_name,'wb')
    dump(clf, clf_model, -1)
    return


# In[11]:


def import_classifier(name):
    """This function import the existing trained model pickle of tagger"""
    classinput=open(name,'rb')
    main_class=load(classinput)
    classinput.close()
    return main_class


# In[12]:


def close_score(guess, truth):
    close = 0
    for i in range(0,len(help_rank_test)):
        if abs(guess[i]-help_rank_test[i])<=1:
            close +=1
    return close/len(help_rank_test)


# In[15]:


# extract the feartures of different reviews based on different corpus
start_time = time.time()
feature_all_words = [(features_extract(d,all_words), c) for (d,c) in data_labled]
feature_content = [(features_extract(d,content_words), c) for (d,c) in data_labled]
feature_negation = [(features_extract(negation(d),all_words), c) for (d,c) in data_labled]
print('Elapsed time:', (time.time() - start_time))


# In[16]:


# split feature and labels
reviews_feature_all_words, help_rank = feature_label(feature_all_words)
reviews_feature_content_words, help_rank = feature_label(feature_content)
reviews_feature_negation_words, help_rank = feature_label(feature_negation)


# In[17]:


# split the data set

# split the feature of all words set
review_train_all_words, review_test_all_words, help_rank_train, help_rank_test = train_test_split(
        reviews_feature_all_words, help_rank, test_size=0.2, random_state=0)

# split the feature of words with content
review_train_content_words, review_test_content_words, help_rank_train, help_rank_test = train_test_split(
        reviews_feature_content_words, help_rank, test_size=0.2, random_state=0)

# split the feature with negation
review_train_negation_words, review_test_negation_words, help_rank_train, help_rank_test = train_test_split(
        reviews_feature_negation_words, help_rank, test_size=0.2, random_state=0)


# In[91]:


# construct bayes model
start_time = time.time()
bayes_model(review_train_all_words,help_rank_train,'bayes-all-words.pkl')
bayes_model(review_train_content_words,help_rank_train,'bayes-content-words.pkl')
bayes_model(review_train_negation_words,help_rank_train,'bayes-negation-words.pkl')
print('Elapsed time:', (time.time() - start_time))


# In[92]:


# construct tree model
start_time = time.time()
tree_model(review_train_all_words,help_rank_train,'tree-all-words.pkl')
tree_model(review_train_content_words,help_rank_train,'tree-content-words.pkl')
tree_model(review_train_negation_words,help_rank_train,'tree-negation-words.pkl')
print('Elapsed time:', (time.time() - start_time))


# In[101]:


# construct cross-valiadaiton model
start_time = time.time()
cross_validation(review_train_all_words,help_rank_train,'cv-all-words.pkl')
cross_validation(review_train_content_words,help_rank_train,'cv-content-words.pkl')
cross_validation(review_train_negation_words,help_rank_train,'cv-negation-words.pkl')
print('Elapsed time:', (time.time() - start_time))


# In[18]:


# import all model that required all words corpus and test accuracy score and close hit.
bayes_all_words_model = import_classifier('bayes-all-words.pkl')
tree_all_words_model = import_classifier('tree-all-words.pkl')
clt_all_words_model = import_classifier('cv-all-words.pkl')


bayes_all_word_score = bayes_all_words_model.score(review_test_all_words, help_rank_test)
bayes_all_word_guess = list(bayes_all_words_model.predict(review_test_all_words))
bayes_all_word_close = close_score(bayes_all_word_guess, help_rank_test)


tree_all_word_score = metrics.accuracy_score(help_rank_test, tree_all_words_model.predict(review_test_all_words))
tree_all_word_guess = list(tree_all_words_model.predict(review_test_all_words))
tree_all_word_close = close_score(tree_all_word_guess, help_rank_test)


clt_all_word_score = clt_all_words_model.score(review_test_all_words, help_rank_test)
clt_all_word_guess = list(clt_all_words_model.predict(review_test_all_words))
clt_all_word_close = close_score(clt_all_word_guess, help_rank_test)


# In[19]:


# import all model that required content words corpus and test accuracy score and close hit.
bayes_content_words_model = import_classifier('bayes-content-words.pkl')
tree_content_words_model = import_classifier('tree-content-words.pkl')
clt_content_words_model = import_classifier('cv-content-words.pkl')

    
bayes_content_word_score = bayes_content_words_model.score(review_test_content_words, help_rank_test)
bayes_content_word_guess = list(bayes_content_words_model.predict(review_test_content_words))
bayes_content_word_close = close_score(bayes_content_word_guess, help_rank_test)

    
tree_content_word_score = metrics.accuracy_score(help_rank_test, tree_content_words_model.predict(review_test_content_words))
tree_content_word_guess = list(tree_content_words_model.predict(review_test_content_words))
tree_content_word_close = close_score(tree_content_word_guess, help_rank_test)
    
    
clt_content_word_score = clt_content_words_model.score(review_test_content_words, help_rank_test)
clt_content_word_guess = list(clt_content_words_model.predict(review_test_content_words))
clt_content_word_close = close_score(clt_content_word_guess, help_rank_test)
    


# In[20]:


# import all model that required negation words corpus and test accuracy score and close hit.

bayes_negation_words_model = import_classifier('bayes-negation-words.pkl')
tree_negation_words_model = import_classifier('tree-negation-words.pkl')
clt_negation_words_model = import_classifier('cv-negation-words.pkl')

bayes_negation_word_score = bayes_negation_words_model.score(review_test_negation_words, help_rank_test)
bayes_negation_word_guess = list(bayes_negation_words_model.predict(review_test_negation_words))
bayes_negation_word_close = close_score(bayes_negation_word_guess, help_rank_test)

tree_negation_word_score = metrics.accuracy_score(help_rank_test, tree_negation_words_model.predict(review_test_negation_words))
tree_negation_word_guess = list(tree_negation_words_model.predict(review_test_negation_words))
tree_negation_word_close = close_score(tree_negation_word_guess, help_rank_test)

clt_negation_word_score = clt_negation_words_model.score(review_test_negation_words, help_rank_test)
clt_negation_word_guess = list(clt_negation_words_model.predict(review_test_negation_words))
clt_negation_word_close = close_score(clt_negation_word_guess, help_rank_test)


# In[21]:


# table creation
x_arix=['bayes_all_words','bayes_content_words','bayes_negation_words','tree_all_words_model',
       'tree_content_words','tree_negation_words','clt_all_words','clt_content_words','clt_negation_word']
score=[bayes_all_word_score,bayes_content_word_score,bayes_negation_word_score,
      tree_all_word_score,tree_content_word_score,tree_negation_word_score,
      clt_all_word_score,clt_content_word_score,clt_negation_word_score]
close=[bayes_all_word_close,bayes_content_word_close,bayes_negation_word_close,
      tree_all_word_close,tree_content_word_close,tree_negation_word_close,
      clt_all_word_close,clt_content_word_close,clt_negation_word_close]
data = {'model name:': x_arix, 'accurary score:': score, 'close hit:': close }


# In[26]:


# plot create
plt.scatter(x='model name:', y='accurary score:', data=data)
plt.xlabel('model name')
plt.ylabel('accurary score')
plt.rcParams["figure.figsize"] = [20,9]
plt.savefig('accuracy figure-new set2.pdf')


# In[27]:


plt.scatter(x='model name:', y='close hit:', data=data)
plt.xlabel('model name')
plt.ylabel('close hit')
plt.rcParams["figure.figsize"] = [20,9]
plt.savefig('close hit-new set2.pdf')


# In[ ]:


######## final model ##########


# In[13]:


data_labled = load_data(1000)
feature_all_words = [(features_extract(d,all_words), c) for (d,c) in data_labled]
reviews_feature_all_words, help_rank = feature_label(feature_all_words)
review_train_all_words, review_test_all_words, help_rank_train, help_rank_test = train_test_split(
        reviews_feature_all_words, help_rank, test_size=0.2, random_state=0)


# In[16]:


start_time = time.time()
tree_model(review_train_all_words,help_rank_train,'tree-final-model.pkl')
print('Elapsed time:', (time.time() - start_time))


# In[14]:



tree_final_model = import_classifier('tree-final-model.pkl')
tree_final_score = tree_final_model.score(review_test_all_words, help_rank_test)
tree_final_guess = list(tree_final_model.predict(review_test_all_words))
tree_final_close = close_score(tree_final_guess, help_rank_test)
print(tree_final_score)
print(tree_final_close)

