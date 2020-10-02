#!/usr/bin/env python
# coding: utf-8

# # Visualize GMM Distributions

# In[1]:


import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import torch
from collections import defaultdict, Counter
import random
import pickle
import itertools

import src.anomaly_model
import src.sent_encoder

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100


# ## Load data

# In[2]:


with open('../data/bnc.pkl', 'rb') as f:
  bnc_sentences = pickle.load(f)

random.seed(12345)
bnc_sentences = random.sample(bnc_sentences, 50)


# In[3]:


enc = src.sent_encoder.SentEncoder()


# ## Plot PCA

# In[4]:


import sklearn.decomposition

#for LAYER in range(13):
for LAYER in [12]:
  tokens, vecs = enc.contextual_token_vecs(bnc_sentences)
  tokens = list(itertools.chain(*tokens))
  vecs = np.vstack(vecs)[:, LAYER, :]
  
  pca = sklearn.decomposition.PCA(n_components=2)
  vecs_pca = pca.fit_transform(vecs)
  vecs_pca_df = pd.DataFrame({'token': tokens, 'x0': vecs_pca[:,0], 'x1': vecs_pca[:,1]})

  plot = sns.scatterplot(data=vecs_pca_df, x='x0', y='x1')
  plt.suptitle(f"Layer: {LAYER}")
  plt.show()


# ## Plot distributions

# In[5]:


with open('../data/bnc.pkl', 'rb') as f:
  bnc_sentences = pickle.load(f)

random.seed(12345)
bnc_sentences = random.sample(bnc_sentences, 1000)


# In[6]:


model = src.anomaly_model.AnomalyModel(bnc_sentences)
all_tokens, all_scores = model.gmm_score(bnc_sentences)


# In[7]:


for layer in range(13):
  layer_scores = np.concatenate([sent_scores[layer] for sent_scores in all_scores])
  sns.histplot(layer_scores, bins=50)
  plt.suptitle(f"Layer: {layer}\nMean:{np.mean(layer_scores):.2f}, STD: {np.std(layer_scores):.2f}")
  plt.show()

