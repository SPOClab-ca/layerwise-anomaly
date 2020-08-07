import os
import random
import pandas as pd

class SentPairGenerator():
  """Methods to generate various pairs of sentences"""

  def __init__(self, data_dir='../data'):
    self.data_dir = data_dir


  def get_wikipedia(self, limit=200):
    """
    Wikipedia, 2016, 10K sentences
    Downloaded from: https://wortschatz.uni-leipzig.de/en/download/english
    """
    with open(os.path.join(self.data_dir, 'leipzig_wikipedia.txt')) as f:
      wiki_sents = f.read().split('\n')[:-1]

    random.seed(12345)
    sentences = []
    for i in range(limit):
      s1 = random.choice(wiki_sents)
      s2 = random.choice(wiki_sents)
      if s1 == s2: continue
      sentences.append((s1, s2))

    return sentences
  

  def get_osterhout_nicol(self, anomaly_type):
    """
    90 sentences from Osterhout and Nicol (1999). Anomaly can be 'syntactic' or 'semantic'.
    """
    df = pd.read_csv(os.path.join(self.data_dir, 'osterhout-nicol.csv'))

    if anomaly_type == 'syntactic':
      df = df[['original_sentence', 'syntactic_anomaly']]
    elif anomaly_type == 'semantic':
      df = df[['original_sentence', 'semantic_anomaly']]
    else:
      assert(False)

    return [tuple(x) for x in df.to_numpy()]
