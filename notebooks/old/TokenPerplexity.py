#!/usr/bin/env python
# coding: utf-8

# # Token-level perplexity
# 
# See whether BERT can find the incorrect token in the sentence.

# In[1]:


from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch


# ## Load RoBERTa

# In[2]:


model_name = 'roberta-base'
bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModelForMaskedLM.from_pretrained(model_name)


# ## Try encoding a sentence

# In[3]:


input_ids = bert_tokenizer.encode("I took my dog for a walking and went to the park.")
prediction_scores = bert_model(torch.tensor(input_ids).unsqueeze(0))[0][0]


# In[4]:


prediction_scores.shape


# In[5]:


for ix in range(len(prediction_scores)):
  print(bert_tokenizer.decode(input_ids[ix]), float(prediction_scores[ix][input_ids[ix]]))


# ## Result
# 
# Yup, "walking" has the lowest score.
