#!/usr/bin/env python
# coding: utf-8

# # Explore properties of Gaussian models / Mahalanobis distance

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
import math
import pickle

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100


# ## Mahalanobis Example
# 
# https://jamesmccaffrey.wordpress.com/2017/11/09/example-of-calculating-the-mahalanobis-distance/
# 
# Note that the covariance matrix doesn't quite match the blog post. The blog post assumes sample covariance matrix (with the n-1 denominator), whereas sklearn uses the simple average. We'll go with sklearn.

# In[2]:


X = np.array([
  [64.0,   580.0,  29.0],
  [66.0,   570.0,  33.0],
  [68.0,   590.0,  37.0],
  [69.0,   660.0,  46.0],
  [73.0,   600.0,  55.0],
])


# In[3]:


mu = X.mean(axis=0)


# In[4]:


mu


# In[5]:


cov = np.cov(X.T, ddof=0)


# In[6]:


cov


# In[7]:


covi = np.linalg.inv(cov)


# In[8]:


covi


# In[9]:


v = np.array([66, 640, 44])


# In[10]:


v - mu


# In[11]:


import scipy.spatial


# In[12]:


dist = scipy.spatial.distance.mahalanobis(v, mu, covi)
dist


# ## Fit same data with GMM

# In[14]:


import sklearn.mixture


# In[15]:


gmm = sklearn.mixture.GaussianMixture()


# In[16]:


gmm.fit(X)


# In[17]:


gmm.score([v])


# ## Connection between the two
# 
# The score returned by GMM is $\log(\frac{1}{(2 \pi)^{D/2} |S|^{1/2}} \exp(-\frac{1}{2}d))$ where $d$ is the Mahalanobis distance.

# In[19]:


math.log(1 / ((2*math.pi)**(3/2) * np.linalg.det(cov)**0.5) * math.exp(-0.5 * dist**2))

