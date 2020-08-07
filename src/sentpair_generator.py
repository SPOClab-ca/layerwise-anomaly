import os
import random

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
