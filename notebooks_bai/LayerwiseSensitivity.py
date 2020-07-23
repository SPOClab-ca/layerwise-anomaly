#!/usr/bin/env python
# coding: utf-8

# # Layerwise Sensitivity
# 
# See how much different layers are sensitive to (1) pairs with dobj/iobj change, and (2) random sentence pairs.

# In[2]:


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

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## Load sentence pairs

# In[3]:


with open('../data/sents.pkl', 'rb') as f:
  data = pickle.load(f)
  data = list(data)


# In[4]:


# https://wortschatz.uni-leipzig.de/en/download/english
# Wikipedia, 2016, 10K sentences
with open('../data/leipzig_wikipedia.txt') as f:
  wiki_sents = f.read().split('\n')[:-1]


# In[5]:


len(data)


# In[11]:


enc = src.sent_encoder.SentEncoder()


# ## Generate boxplots

# In[ ]:


distances = []
for layer in range(13):
  print('Processing layer:', layer)
  for sent_pair in data:
    dist = np.linalg.norm(enc.evaluate_contextual_diff(sent_pair, layer=layer).cpu().detach().numpy())
    distances.append(pd.Series({
      'layer': layer,
      'dist': dist,
      'sent1': sent_pair[0],
      'sent2': sent_pair[1],
    }))
distances = pd.DataFrame(distances)


# In[23]:


model_name = 'roberta-base'
sns.boxplot(x='layer', y='dist', data=distances)
plt.title(f"Sentence embedding distance for dobj/iobj pairs.\nmodel: {model_name}")
plt.ylim(0)
plt.show()


# In[22]:


wiki_distances = []
for layer in range(13):
  print('Processing layer:', layer)
  for i in range(300):
    s1 = random.choice(wiki_sents)
    s2 = random.choice(wiki_sents)
    if s1 == s2: continue
    dist = np.linalg.norm(enc.evaluate_contextual_diff((s1, s2), layer=layer).cpu().detach().numpy())
    wiki_distances.append(pd.Series({
      'layer': layer,
      'dist': dist,
      'sent1': sent_pair[0],
      'sent2': sent_pair[1],
    }))
wiki_distances = pd.DataFrame(wiki_distances)


# In[24]:


sns.boxplot(x='layer', y='dist', data=wiki_distances)
plt.title(f"Sentence embedding distance for wiki pairs.\nmodel: {model_name}")
plt.ylim(0)
plt.show()


# ## Combined plot

# In[32]:


distances['dataset'] = 'dobj/iobj'
wiki_distances['dataset'] = 'random wiki'
combined_df = pd.concat([distances, wiki_distances])


# In[33]:


sns.barplot(x='layer', y='dist', hue='dataset', data=combined_df)
plt.title(f"model: {model_name}")
plt.ylim(0)
plt.show()

