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

# For the ba-construction, verb must have a state-changing effect on the object.
CHINESE_TRANSITIVE_VERBS = """感动 骗 骂 吓 打 害 震惊 忘 揍 救 坑 气 蒙 忘记 抓住 杀 举报
通知 联络 帮 回复 采访 放 投诉 警告 批评""".split()

CHINESE_HUMAN_NOUNS = """工程师 留学生 医生 律师 老师 农民 作家 教师 记者 编辑 基督徒 学生
设计师 会计 科学家 公务员 博士 护士 学者 大学生 教授 演员 艺术家 党员 人 移民 警察 司机 工人
企业家 残疾人 保安 歌手 专家 高手 创业者 男人 女人 女孩 孩子 外国人 老人""".split()


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


  def get_paws(self, is_paraphrase):
    """
    Sentences from PAWS-X (https://github.com/google-research-datasets/paws/tree/master/pawsx)
    Label = 1 if paraphrase, 0 if not paraphrase
    """
    df = pd.read_csv(os.path.join(self.data_dir, 'paws-x-dev.tsv'), delimiter='\t')

    if is_paraphrase:
      df = df[df.label == 1]
    else:
      df = df[df.label == 0]

    df = df[['sentence1', 'sentence2']]
    return [tuple(x) for x in df.to_numpy()]
  

  def get_chinese_news(self, limit=200):
    with open(os.path.join(self.data_dir, 'chinese-news.txt')) as f:
      sents = f.read().split('\n')[:-1]

    random.seed(12345)
    sentences = []
    for i in range(limit):
      s1 = random.choice(sents)
      s2 = random.choice(sents)
      if s1 == s2: continue
      sentences.append((s1, s2))

    return sentences


  def get_chinese_ba_alternation(self, limit=200):
    random.seed(12345)
    sentences = []
    for i in range(limit):
      vb = random.choice(CHINESE_TRANSITIVE_VERBS)
      subj = random.choice(CHINESE_HUMAN_NOUNS)
      obj = random.choice(CHINESE_HUMAN_NOUNS)
      if subj == obj:
        continue

      s1 = f"{subj}{vb}了{obj}。"
      s2 = f"{subj}把{obj}{vb}了。"
      sentences.append((s1, s2))
    return sentences
