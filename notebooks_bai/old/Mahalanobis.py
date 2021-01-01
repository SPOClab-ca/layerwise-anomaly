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

# In[13]:


import sklearn.mixture


# In[14]:


gmm = sklearn.mixture.GaussianMixture()


# In[15]:


gmm.fit(X)


# In[16]:


gmm.score([v])


# ## Connection between the two
# 
# The score returned by GMM is $G = \log(\frac{1}{(2 \pi)^{D/2} |S|^{1/2}} \exp(-\frac{1}{2}d^2))$ where $d$ is the Mahalanobis distance.
# 
# Equivalently, $d^2 = -D \log(2 \pi) - \log |S| - 2G$.
# 
# Our method of summing `gmm.score_samples` across tokens can be viewed as:
# 1. Joint log likelihood of all the tokens: $\sum_{i=1}^n G_i = \log(P(w_i) \cdots P(w_n))$
# 2. Sum of squared Mahalanobis distances: $\sum_{i=1}^n G_i = -n (\frac{D}{2} \log (2 \pi) - \frac{1}{2} \log|S|)) - \frac{1}{2} \sum_{i=1}^n d^2$
# 
# This is to show that what we're doing is theoretically justified. We have empirical support as well, since this method performs better than several other methods by BLiMP accuracy.

# In[17]:


math.log(1 / ((2*math.pi)**(3/2) * np.linalg.det(cov)**0.5) * math.exp(-0.5 * dist**2))


# In[19]:


# Inverse formula
math.sqrt(-3 * math.log(2*math.pi) - math.log(scipy.linalg.det(gmm.covariances_[0])) - 2*gmm.score([v]))

