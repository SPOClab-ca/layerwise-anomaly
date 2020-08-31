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

# In[13]:


model = src.anomaly_model.AnomalyModel(bnc_sentences)


# In[18]:


def all_layer_scores(sent):
  tokens, all_layer = model.gmm_score(sent)
  plt.figure(figsize=(8, 8))
  plt.imshow(all_layer, origin='lower')
  plt.xticks(range(len(tokens)), tokens, rotation='vertical')
  plt.yticks(range(12), range(12))
  plt.ylabel('Layer')
  plt.show()

all_layer_scores("The cats won't eating the food that Mary gives them.")


# ## Evaluate on Osterhout / Nicol data

# In[15]:


sentgen = src.sentpair_generator.SentPairGenerator()


# In[22]:


for layer in range(13):
  syn_score = model.eval_sent_pairs(sentgen.get_osterhout_nicol(anomaly_type='syntactic'), layer)
  sem_score = model.eval_sent_pairs(sentgen.get_osterhout_nicol(anomaly_type='semantic'), layer)
  print(layer, syn_score, sem_score)

