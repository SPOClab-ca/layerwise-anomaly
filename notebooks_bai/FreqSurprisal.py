#!/usr/bin/env python
# coding: utf-8

# # Frequency and Surprisal
# 
# How sensitive are all the different layers to token frequency?

# In[ ]:


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

import src.anomaly_model

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## Load model

# In[ ]:


with open('../data/bnc.pkl', 'rb') as f:
  bnc_sentences = pickle.load(f)

random.seed(12345)
bnc_sentences_train = random.sample(bnc_sentences, 5000)
bnc_sentences_test = random.sample(bnc_sentences, 5000)


# In[ ]:


model = src.anomaly_model.AnomalyModel(bnc_sentences_train, model_name='xlnet-base-cased')


# ## Tabulate token frequencies

# In[ ]:


tokens, all_layer = model.gmm_score(bnc_sentences_test)


# In[ ]:


freq_counter = Counter(itertools.chain.from_iterable(tokens))


# In[ ]:


freq_counter.most_common(10)


# ## Plot all layers

# In[ ]:


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

