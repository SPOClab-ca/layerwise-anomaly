#!/usr/bin/env python
# coding: utf-8

# # GMM Anomaly Detection in contextual tokens

# In[1]:


import sys
sys.path.append('../')

import pickle
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import random
import sklearn.mixture

import src.sent_encoder
import src.sentpair_generator

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


model_name = 'roberta-base'
bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModelForMaskedLM.from_pretrained(model_name)


# ## Pick random subset of sentences

# In[3]:


with open('../data/bnc.pkl', 'rb') as f:
  bnc_sentences = pickle.load(f)

random.seed(12345)
bnc_sentences = random.sample(bnc_sentences, 1000)


# ## Feed them through BERT

# In[4]:


enc = src.sent_encoder.SentEncoder()


# In[5]:


bnc_vecs = enc.contextual_token_vecs(bnc_sentences, layer=-2)


# ## Train GMM

# In[6]:


gmm = sklearn.mixture.GaussianMixture()
gmm.fit(bnc_vecs)


# In[7]:


def infer_new_sentence(sent):
  ids = [x for x in enc.auto_tokenizer(sent)['input_ids'] if x not in enc.auto_tokenizer.all_special_ids]
  sent_vecs = enc.contextual_token_vecs([sent])
  assert len(ids) == sent_vecs.shape[0]
  
  for i in range(sent_vecs.shape[0]):
    print(enc.auto_tokenizer.decode(ids[i]), gmm.score([sent_vecs[i]]))


# In[8]:


infer_new_sentence("The student laughs.")


# In[9]:


infer_new_sentence("The student laugh.")


# ## Evaluate on Osterhout / Nicol data

# In[10]:


sentgen = src.sentpair_generator.SentPairGenerator()


# In[11]:


def gmm_score(sent):
  ids = [x for x in enc.auto_tokenizer(sent)['input_ids'] if x not in enc.auto_tokenizer.all_special_ids]
  sent_vecs = enc.contextual_token_vecs([sent])
  assert len(ids) == sent_vecs.shape[0]
  
  score = 0
  for i in range(sent_vecs.shape[0]):
    score += gmm.score([sent_vecs[i]])
  return score
    
    
# Assume first of pair is the correct one
def eval_sent_pairs(sentpairs):
  got_right = 0
  for correct_sent, incorrect_sent in sentpairs:
    if gmm_score(correct_sent) > gmm_score(incorrect_sent):
      got_right += 1
  print(got_right / len(sentpairs))


# In[12]:


eval_sent_pairs(sentgen.get_osterhout_nicol(anomaly_type='syntactic'))


# In[13]:


eval_sent_pairs(sentgen.get_osterhout_nicol(anomaly_type='semantic'))

