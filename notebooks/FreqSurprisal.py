#!/usr/bin/env python
# coding: utf-8

# # Frequency and Surprisal
# 
# How sensitive are all the different layers to token frequency?

# In[1]:


import sys
sys.path.append('../')

import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import seaborn as sns
import itertools
import math
import scipy
import sklearn

import src.anomaly_model
import src.sent_encoder

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## Load model

# In[2]:


with open('../data/bnc.pkl', 'rb') as f:
  bnc_sentences = pickle.load(f)

random.seed(12345)
bnc_sentences_train = random.sample(bnc_sentences, 5000)
bnc_sentences_test = random.sample(bnc_sentences, 5000)


# In[ ]:


model = src.anomaly_model.AnomalyModel(bnc_sentences_train, model_name='roberta-base')


# ## Tabulate token frequencies

# In[ ]:


tokens, all_layer = model.gmm_score(bnc_sentences_test)


# In[ ]:


freq_counter = Counter(itertools.chain.from_iterable(tokens))


# In[ ]:


freq_counter.most_common(10)


# ## Plot frequency vs anomaly scores (Figure 3 in paper)

# In[7]:


def plot_for_one_layer(layer):
  df = []
  for sent_ix, sent in enumerate(tokens):
    for tok_ix, token in enumerate(sent):
      surprisal = all_layer[sent_ix][layer, tok_ix]
      logfreq = math.log(freq_counter[token])
      df.append({'token': token, 'logfreq': logfreq, 'surprisal': surprisal})
  df = pd.DataFrame(df)
  corr = scipy.stats.pearsonr(df.logfreq, df.surprisal)[0]
  print(corr)

  #sns.regplot(x='logfreq', y='surprisal', data=df)
  #plt.title(f'Layer {layer}, pearson={corr:0.2f}')
  #plt.ylabel('Anomaly Score')
  #plt.xlabel('Log Frequency')
  #plt.show()


# In[ ]:


for layer in range(13):
  plot_for_one_layer(layer)


# ## Plot PCA/t-SNE of rare tokens (not in paper)

# In[3]:


enc = src.sent_encoder.SentEncoder()
all_tokens, all_vecs = enc.contextual_token_vecs(random.sample(bnc_sentences_test, 1000))
freq_counter = Counter(itertools.chain.from_iterable(all_tokens))


# In[20]:


layer = 1
pca_model = sklearn.decomposition.PCA(n_components=50)
pca_vecs = pca_model.fit_transform(np.vstack(all_vecs)[:, layer, :])


# In[ ]:


tsne_model = sklearn.manifold.TSNE(n_components=2, n_jobs=-1, verbose=10)
tsne_vecs = tsne_model.fit_transform(pca_vecs)


# In[22]:


df = []
combined_ix = 0
for sent_ix, sent in enumerate(all_tokens):
  for tok_ix, token in enumerate(sent):
    logfreq = math.log(freq_counter[token])
    dim1 = tsne_vecs[combined_ix, 0]
    dim2 = tsne_vecs[combined_ix, 1]
    df.append({'token': token, 'logfreq': logfreq, 'dim1': dim1, 'dim2': dim2})
    combined_ix += 1
df = pd.DataFrame(df)


# In[23]:


df['rare'] = df.logfreq < df.logfreq.quantile(0.2)


# In[24]:


sns.scatterplot(x=df.dim1, y=df.dim2, hue=df.rare)
plt.show()
