#!/usr/bin/env python
# coding: utf-8

# # Perplexity with GPT models

# In[1]:


from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import matplotlib.pyplot as plt


# In[ ]:


model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')


# In[3]:


def get_perplexity(sent):
  input_ids = torch.tensor(tokenizer.encode(sent))
  with torch.no_grad():
    perplexity = model(input_ids, labels=input_ids)[0] * len(input_ids)
  return float(perplexity)


# In[4]:


get_perplexity("Colorless green ideas sleep furiously.")


# In[5]:


get_perplexity("Furiously sleep ideas green colorless.")


# ## Attention heads

# In[ ]:


input_ids = torch.tensor(tokenizer.encode("Colorless green ideas sleep furiously."))
input_labels = tokenizer.convert_ids_to_tokens(input_ids)

layer = 6
for attn_head_id in range(12):
  with torch.no_grad():
    A = model(input_ids, labels=input_ids, output_attentions=True)[3][layer][attn_head_id]
  plt.imshow(A)
  plt.xticks(range(len(input_labels)), input_labels, rotation='vertical')
  plt.yticks(range(len(input_labels)), input_labels)
  plt.title(f'Layer: {layer}    Head: {attn_head_id}')
  plt.show()

