#!/usr/bin/env python
# coding: utf-8

# # Layerwise Sensitivity
# 
# See how much different layers are sensitive to (1) pairs with dobj/iobj change, and (2) random sentence pairs.

# In[1]:


import sys
sys.path.append('../')

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd
import os, sys, time, re
import matplotlib.pyplot as plt
import random
import pickle
from scipy.spatial.distance import cosine
import seaborn as sns

import src.sent_encoder
import src.sentpair_generator

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## Load sentence pairs

# In[2]:


with open('../data/sents.pkl', 'rb') as f:
  data = pickle.load(f)
  data = list(data)


# In[3]:


# https://wortschatz.uni-leipzig.de/en/download/english
# Wikipedia, 2016, 10K sentences
sentgen = src.sentpair_generator.SentPairGenerator()
wiki_sents = sentgen.get_wikipedia()


# In[4]:


len(data)


# In[5]:


enc = src.sent_encoder.SentEncoder()


# ## Generate boxplots
distances = enc.get_layer_distance_df(data)
# In[ ]:


model_name = 'roberta-base'
sns.boxplot(x='layer', y='dist', data=distances)
plt.title(f"Sentence embedding distance for dobj/iobj pairs.\nmodel: {model_name}")
plt.ylim(0)
plt.show()


# In[6]:


wiki_sents = sentgen.get_wikipedia()
wiki_distances = enc.get_layer_distance_df(wiki_sents)


# In[7]:


sns.boxplot(x='layer', y='dist', data=wiki_distances)
plt.title(f"Sentence embedding distance for wiki pairs.\nmodel: {model_name}")
plt.ylim(0)
plt.show()


# ## Combined plot

# In[ ]:


distances['dataset'] = 'dobj/iobj'
wiki_distances['dataset'] = 'random wiki'
combined_df = pd.concat([distances, wiki_distances])


# In[ ]:


sns.barplot(x='layer', y='dist', hue='dataset', data=combined_df)
plt.title(f"model: {model_name}")
plt.ylim(0)
plt.show()

