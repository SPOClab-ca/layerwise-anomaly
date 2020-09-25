#!/usr/bin/env python
# coding: utf-8

# # GMM Anomaly Detection in contextual tokens

# In[1]:


import sys
sys.path.append('../')

import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

import src.sentpair_generator
import src.anomaly_model

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## Pick random subset of sentences

# In[2]:


with open('../data/bnc.pkl', 'rb') as f:
  bnc_sentences = pickle.load(f)

random.seed(12345)
bnc_sentences = random.sample(bnc_sentences, 1000)


# ## Plot of GMM score at each layer and word

# In[3]:


model = src.anomaly_model.AnomalyModel(bnc_sentences)


# In[4]:


def all_layer_scores(sent):
  tokens, all_layer = model.gmm_score([sent])
  tokens = tokens[0]
  all_layer = all_layer[0]
  plt.figure(figsize=(8, 8))
  plt.imshow(all_layer, origin='lower')
  plt.xticks(range(len(tokens)), tokens, rotation='vertical')
  plt.yticks(range(12), range(12))
  plt.ylabel('Layer')
  for (j,i),label in np.ndenumerate(all_layer):
    plt.text(i, j, int(label), ha='center', va='center', color='white')
  plt.show()

all_layer_scores("The cats won't eating the food that Mary gives them.")


# In[ ]:


all_layer_scores("Corey's hamster entertained a nearby backpack and filled it with sawdust.")


# ## Evaluate on Osterhout / Nicol data

# In[5]:


sentgen = src.sentpair_generator.SentPairGenerator()


# In[18]:


def process_sentpair_dataset(csvname, correct_col, wrong_col):
  scores = []
  for layer in range(13):
    results = model.eval_sent_pairs(sentgen.get_csv_based_dataset(csvname, correct_col, wrong_col), layer)
    score = sum(results) / len(results)
    scores.append(score)
    print(layer, score)
    
  plt.plot(scores)
  plt.ylim((0, 1))
  plt.xticks(range(0, 13))
  plt.title(f"{csvname}: {correct_col} vs {wrong_col}")
  plt.xlabel('Layer')
  plt.ylabel('GMM Accuracy')
  plt.show()


# In[19]:


process_sentpair_dataset('osterhout-nicol.csv', 'original_sentence', 'syntactic_anomaly')


# In[20]:


process_sentpair_dataset('osterhout-nicol.csv', 'original_sentence', 'semantic_anomaly')


# ## Evaluate on Pylkkanen data

# In[21]:


process_sentpair_dataset('pylkkanen.csv', 'sent_control', 'sent_coercion')


# In[22]:


process_sentpair_dataset('pylkkanen.csv', 'sent_control', 'sent_anomaly')


# ## Evaluate on Warren data

# In[23]:


process_sentpair_dataset('warren.csv', 'sent_control', 'sent_no_violation')


# In[24]:


process_sentpair_dataset('warren.csv', 'sent_control', 'sent_violation')

