#!/usr/bin/env python
# coding: utf-8

# # Performance of MLM
# 
# What is the performance of all of these tasks using MLM instead of GMM?

# In[35]:


import sys
sys.path.append('../')

import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import transformers
from transformers import AutoTokenizer

import src.sentpair_generator
import src.anomaly_model

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100


# ## Train model

# In[28]:


with open('../data/bnc.pkl', 'rb') as f:
  bnc_sentences = pickle.load(f)

random.seed(12345)
bnc_sentences = random.sample(bnc_sentences, 1000)


# In[29]:


model = src.anomaly_model.AnomalyModel(bnc_sentences, model_name='xlnet-base-cased')


# In[30]:


sentgen = src.sentpair_generator.SentPairGenerator()


# In[31]:


for task_name, sent_pair_set in sentgen.get_hand_selected().items():
  print(task_name, len(sent_pair_set.sent_pairs))


# ## Filter sentences that are in all of their vocab

# In[36]:


tok_roberta = AutoTokenizer.from_pretrained('roberta-base')
tok_bert = AutoTokenizer.from_pretrained('bert-base-uncased')
tok_xlnet = AutoTokenizer.from_pretrained('xlnet-base-cased')


# In[32]:


# Return true if the list of tokens differs in exactly one place
def is_single_diff(toks1, toks2):
  if len(toks1) != len(toks2):
    return False
  
  diff_toks = 0
  for ix in range(len(toks1)):
    if toks1[ix] != toks2[ix]:
      diff_toks += 1
  
  return diff_toks == 1


# In[37]:


def works_for_model(tokenizer, sent1, sent2):
  toks1 = tokenizer.tokenize(sent1)
  toks2 = tokenizer.tokenize(sent2)
  return is_single_diff(toks1, toks2)


# In[38]:


for task_name, sent_pair_set in sentgen.get_hand_selected().items():
  num_include = 0
  for sent1, sent2 in sent_pair_set.sent_pairs:
    if works_for_model(tok_roberta, sent1, sent2) and         works_for_model(tok_bert, sent1, sent2) and        works_for_model(tok_xlnet, sent1, sent2):
      num_include += 1
  print(task_name, num_include)

