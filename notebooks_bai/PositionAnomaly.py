#!/usr/bin/env python
# coding: utf-8

# # Position Anomaly Difference
# 
# Is the anomaly concentrated on the word that's wrong, or distributed throughout?

# In[1]:


import sys
sys.path.append('../')

import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import src.sentpair_generator
import src.anomaly_model

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100


# ## Train model

# In[2]:


with open('../data/bnc.pkl', 'rb') as f:
  bnc_sentences = pickle.load(f)

random.seed(12345)
bnc_sentences = random.sample(bnc_sentences, 1000)


# In[ ]:


model = src.anomaly_model.AnomalyModel(bnc_sentences)


# In[3]:


sentgen = src.sentpair_generator.SentPairGenerator()


# ## Pick random sentences
# 
# Create a dataset consisting of max 100 sentences from each type, excluding pragmatic ones

# In[10]:


for task_name, sent_pair_set in sentgen.get_hand_selected().items():
  print(task_name, len(sent_pair_set.sent_pairs))


# In[7]:


sent_pairs = []
for _, sent_pair_set in sentgen.get_hand_selected().items():
  if sent_pair_set.category != 'Pragmatic':
    cur_sent_pairs = sent_pair_set.sent_pairs
    if len(cur_sent_pairs) > 100:
      cur_sent_pairs = random.sample(cur_sent_pairs, 100)
    sent_pairs.extend(cur_sent_pairs)


# In[11]:


print(len(sent_pairs))

