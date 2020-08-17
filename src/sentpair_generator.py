import os
import random
import pandas as pd
import pickle


TRANSITIVE_VERBS = """knew killed attacked asked confronted approached thanked called saw
trusted hugged liked followed interrupted begged visited warned rejected helped hated
criticized wanted missed protected punched""".split()

HUMAN_NOUNS = """man woman boy girl teacher student thief child kid soldier robot artist
protester president drummer zombie baby teenager captain prisoner engineer pilot waiter
doctor landlord manager worker victim employee painter priest""".split()


class SentPairGenerator():
  """Methods to generate various pairs of sentences"""

  def __init__(self, data_dir='../data'):
    self.data_dir = data_dir
    with open(os.path.join(self.data_dir, 'leipzig_wikipedia.txt')) as f:
      self.wiki_sents = f.read().split('\n')[:-1]


  def get_wikipedia(self, limit=200):
    """
    Wikipedia, 2016, 10K sentences
    Downloaded from: https://wortschatz.uni-leipzig.de/en/download/english
    """
    random.seed(12345)
    sentences = []
    for i in range(limit):
      s1 = random.choice(self.wiki_sents)
      s2 = random.choice(self.wiki_sents)
      if s1 == s2: continue
      sentences.append((s1, s2))

    return sentences
  

  def get_wikipedia_delete_random_word(self, limit=200):
    random.seed(12345)
    sentences = []
    for i in range(limit):
      s1 = random.choice(self.wiki_sents).split()
      s2 = list(s1)
      s2.pop(random.randrange(len(s2)))
      sentences.append((' '.join(s1), ' '.join(s2)))
    return sentences
  

  def get_dative_alternation(self):
    with open('../data/dative-alternation.pkl', 'rb') as f:
      data = pickle.load(f)
      data = list(data)
    return data
  

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


  def get_transitive_swap_subject(self, limit=200):
    """Swap the subject with a random other word."""
    random.seed(12345)
    sentences = []
    for i in range(limit):
      vb = random.choice(TRANSITIVE_VERBS)
      subj1 = random.choice(HUMAN_NOUNS)
      subj2 = random.choice(HUMAN_NOUNS)
      obj = random.choice(HUMAN_NOUNS)
      if subj1 == subj2 or subj1 == obj or subj2 == obj:
        continue

      s1 = f"The {subj1} {vb} the {obj}."
      s2 = f"The {subj2} {vb} the {obj}."
      sentences.append((s1, s2))
    return sentences


  def get_transitive_swap_subject_object(self, limit=200):
    """Swap the subject with the object."""
    random.seed(12345)
    sentences = []
    for i in range(limit):
      vb = random.choice(TRANSITIVE_VERBS)
      subj = random.choice(HUMAN_NOUNS)
      obj = random.choice(HUMAN_NOUNS)
      if subj == obj:
        continue

      s1 = f"The {subj} {vb} the {obj}."
      s2 = f"The {obj} {vb} the {subj}."
      sentences.append((s1, s2))
    return sentences


  def get_transitive_replace_determiner(self, limit=200):
    """Swap the determiner of the object from 'the' to 'a'."""
    random.seed(12345)
    sentences = []
    for i in range(limit):
      vb = random.choice(TRANSITIVE_VERBS)
      subj = random.choice(HUMAN_NOUNS)
      obj = random.choice(HUMAN_NOUNS)
      if subj == obj:
        continue

      s1 = f"The {subj} {vb} the {obj}."
      s2 = f"The {subj} {vb} a {obj}."
      sentences.append((s1, s2))
    return sentences


  def get_sts_2012(self, similarity_level):
    """
    Sentences from SentEval 2012, train split, video caption portion. Similarity level
    is on 0-5 scale, but consider 'low' to be <= 1, 'high' to be >= 4.
    """
    df = pd.read_csv(os.path.join(self.data_dir, 'sts12-vid.csv'))

    if similarity_level == 'low':
      df = df[df.sim <= 1]
    elif similarity_level == 'high':
      df = df[df.sim >= 4]
    else:
      assert(False)

    df = df[['sent1', 'sent2']]
    return [tuple(x) for x in df.to_numpy()]
