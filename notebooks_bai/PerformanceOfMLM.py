#!/usr/bin/env python
# coding: utf-8

# # Performance of MLM
# 
# What is the performance of all of these tasks using MLM instead of GMM?

# In[1]:


import sys
sys.path.append('../')

import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

import torch
import transformers
from transformers import AutoTokenizer, XLNetLMHeadModel

import src.sentpair_generator
import src.anomaly_model

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100


# ## Load sentences

# In[2]:


sentgen = src.sentpair_generator.SentPairGenerator()


# ## Filter sentences that are in all of their vocab

# In[11]:


tok_roberta = AutoTokenizer.from_pretrained('roberta-base')
tok_bert = AutoTokenizer.from_pretrained('bert-base-uncased')
tok_xlnet = AutoTokenizer.from_pretrained('xlnet-base-cased')


# In[12]:


# Return true if the list of tokens differs in exactly one place
def is_single_diff(toks1, toks2):
  if len(toks1) != len(toks2):
    return False
  
  diff_toks = 0
  for ix in range(len(toks1)):
    if toks1[ix] != toks2[ix]:
      diff_toks += 1
  
  return diff_toks == 1


# In[13]:


def works_for_model(tokenizer, sent1, sent2):
  toks1 = tokenizer.tokenize(sent1)
  toks2 = tokenizer.tokenize(sent2)
  return is_single_diff(toks1, toks2)


# In[14]:


sent_pairs = defaultdict(list)
for task_name, sent_pair_set in sentgen.get_hand_selected().items():
  for sent1, sent2 in sent_pair_set.sent_pairs:
    if works_for_model(tok_roberta, sent1, sent2) and         works_for_model(tok_bert, sent1, sent2) and        works_for_model(tok_xlnet, sent1, sent2):
      sent_pairs[task_name].append((sent1, sent2))


# In[15]:


for task_name, sent_pair_set in sent_pairs.items():
  print(task_name, len(sent_pair_set))


# ## Fill Mask Accuracy

# In[ ]:


from transformers import pipeline
nlp = pipeline("fill-mask", model='bert-base-uncased')


# In[ ]:


def fill_one(sent1, sent2):
  toks1 = nlp.tokenizer(sent1, add_special_tokens=False)['input_ids']
  toks2 = nlp.tokenizer(sent2, add_special_tokens=False)['input_ids']

  masked_toks = []
  dtok1 = None
  dtok2 = None
  for ix in range(len(toks1)):
    if toks1[ix] != toks2[ix]:
      masked_toks.append(nlp.tokenizer.mask_token_id)
      dtok1 = toks1[ix]
      dtok2 = toks2[ix]
    else:
      masked_toks.append(toks1[ix])

  res = nlp(nlp.tokenizer.decode(masked_toks), targets=[nlp.tokenizer.decode(dtok1), nlp.tokenizer.decode(dtok2)])
  return res[0]['token'] == dtok1


# In[ ]:


def mlm_accuracy(sentpairs):
  res = [fill_one(s1, s2) for (s1, s2) in sentpairs]
  return sum(res) / len(sentpairs)

for task_name, sents in sent_pairs.items():
  print(task_name, mlm_accuracy(sents))


# ## XLNet needs to be done differently

# In[7]:


model_name = 'xlnet-base-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = XLNetLMHeadModel.from_pretrained(model_name)


# In[8]:


def fill_one(sent1, sent2):
  toks1 = tokenizer(sent1, add_special_tokens=False)['input_ids']
  toks2 = tokenizer(sent2, add_special_tokens=False)['input_ids']

  masked_toks = []
  masked_ix = None
  dtok1 = None
  dtok2 = None
  for ix in range(len(toks1)):
    if toks1[ix] != toks2[ix]:
      masked_toks.append(tokenizer.mask_token_id)
      masked_ix = ix
      dtok1 = toks1[ix]
      dtok2 = toks2[ix]
    else:
      masked_toks.append(toks1[ix])

  logit1 = model(torch.tensor([masked_toks])).logits[0, masked_ix, dtok1]
  logit2 = model(torch.tensor([masked_toks])).logits[0, masked_ix, dtok2]
  return bool(logit1 > logit2)


# In[ ]:


def mlm_accuracy(sentpairs):
  res = [fill_one(s1, s2) for (s1, s2) in sentpairs]
  return sum(res) / len(sentpairs)

for task_name, sents in sent_pairs.items():
  print(task_name, mlm_accuracy(sents))


# ## Try using Gaussian model on same data, use best layer for each model

# In[17]:


with open('../data/bnc.pkl', 'rb') as f:
  bnc_sentences = pickle.load(f)

random.seed(12345)
bnc_sentences = random.sample(bnc_sentences, 1000)


# In[57]:


MODEL_NAME = 'bert-base-uncased'
MODEL_LAYER = 9

model = src.anomaly_model.AnomalyModel(bnc_sentences, model_name=MODEL_NAME)


# In[58]:


def process_sentpair_dataset(taskname, sent_pairs):
  scores = []
  for layer in [MODEL_LAYER]:
    results = model.eval_sent_pairs(sent_pairs, layer)
    scores.extend([{'taskname': taskname, 'layer': layer, 'score': r} for r in results])
  scores = pd.DataFrame(scores)
  return scores


# In[59]:


all_scores = []
for taskname, sentpairs in sent_pairs.items():
  task_scores = process_sentpair_dataset(taskname, sentpairs)
  all_scores.append(task_scores)
  
# Role-88 is special...
#taskname = 'ROLE-88'
#sentpairs = sentgen.get_hand_selected()['ROLE-88']
#task_scores = process_sentpair_dataset(taskname, sentpairs.sent_pairs)
#all_scores.append(task_scores)
  
all_scores = pd.concat(all_scores)


# In[60]:


all_scores['Correct'] = all_scores.score > 0


# In[61]:


all_scores[['taskname', 'Correct']].groupby('taskname', sort=False).mean()

