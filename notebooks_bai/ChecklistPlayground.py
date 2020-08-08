#!/usr/bin/env python
# coding: utf-8

# # CheckList Playground
# 
# Useful for coming up with a lot of words of some class (eg: transitive verb).

# In[1]:


import checklist
from checklist.editor import Editor


# In[2]:


editor = Editor()


# In[9]:


editor.suggest('The {mask} showed him his car.')

