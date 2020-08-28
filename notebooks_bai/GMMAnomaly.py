#!/usr/bin/env python
# coding: utf-8

# # GMM Anomaly Detection in contextual tokens

# In[1]:


import sys
sys.path.append('../')

import pickle
import random

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


# ## Train anomaly GMM

# In[3]:


anom = src.anomaly_model.AnomalyModel(bnc_sentences, layer=-2)


# In[4]:


anom.gmm_score("The student laughs.", verbose=True)


# In[5]:


anom.gmm_score("The student laugh.", verbose=True)


# ## Evaluate on Osterhout / Nicol data

# In[6]:


sentgen = src.sentpair_generator.SentPairGenerator()


# In[7]:


anom.eval_sent_pairs(sentgen.get_osterhout_nicol(anomaly_type='syntactic'))


# In[8]:


anom.eval_sent_pairs(sentgen.get_osterhout_nicol(anomaly_type='semantic'))

