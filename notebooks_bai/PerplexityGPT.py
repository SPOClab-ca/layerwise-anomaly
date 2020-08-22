#!/usr/bin/env python
# coding: utf-8

# # Perplexity with GPT models

# In[1]:


from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch


# In[ ]:


model = GPT2LMHeadModel.from_pretrained('gpt2').cuda()
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')


# In[3]:


def get_perplexity(sent):
  input_ids = torch.tensor(tokenizer.encode(sent)).cuda()
  with torch.no_grad():
    perplexity = model(input_ids, labels=input_ids)[0] * len(input_ids)
  return float(perplexity)


# In[4]:


get_perplexity("Colorless green ideas sleep furiously.")


# In[5]:


get_perplexity("Furiously sleep ideas green colorless.")

