"""
Evaluate the accuracy of MLM on all tasks using specified LM

Example usage:
PYTHONPATH=. time python scripts/mlm_accuracy.py \
  model_name=roberta-base
"""
import argparse
from collections import defaultdict
import torch
import transformers
from transformers import AutoTokenizer
from transformers import pipeline

import src.sentpair_generator

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='roberta-base')
args = parser.parse_args()


# Load sentences
sentgen = src.sentpair_generator.SentPairGenerator(data_dir='./data')


# Filter sentences that are in all of their vocab
tok_roberta = AutoTokenizer.from_pretrained('roberta-base')
tok_bert = AutoTokenizer.from_pretrained('bert-base-uncased')
tok_xlnet = AutoTokenizer.from_pretrained('xlnet-base-cased')


# Return true if the list of tokens differs in exactly one place
def is_single_diff(toks1, toks2):
  if len(toks1) != len(toks2):
    return False
  
  diff_toks = 0
  for ix in range(len(toks1)):
    if toks1[ix] != toks2[ix]:
      diff_toks += 1
  
  return diff_toks == 1


def works_for_model(tokenizer, sent1, sent2):
  toks1 = tokenizer.tokenize(sent1)
  toks2 = tokenizer.tokenize(sent2)
  return is_single_diff(toks1, toks2)


sent_pairs = defaultdict(list)
for task_name, sent_pair_set in sentgen.get_hand_selected().items():
  for sent1, sent2 in sent_pair_set.sent_pairs:
    if works_for_model(tok_roberta, sent1, sent2) and \
        works_for_model(tok_bert, sent1, sent2) and \
        works_for_model(tok_xlnet, sent1, sent2):
      sent_pairs[task_name].append((sent1, sent2))


# Fill Mask Accuracy
nlp = pipeline("fill-mask", model=args.model_name)


def fill_one(sent1, sent2):
  toks1 = nlp.tokenizer(sent1, add_special_tokens=False)['input_ids']
  toks2 = nlp.tokenizer(sent2, add_special_tokens=False)['input_ids']

  masked_toks = []
  dtok1 = None
  dtok2 = None
  for ix in range(len(toks1)):
    if toks1[ix] != toks2[ix]:
      masked_toks.append(nlp.tokenizer.mask_token_id)
      dtok1 = toks1[ix]
      dtok2 = toks2[ix]
    else:
      masked_toks.append(toks1[ix])

  res = nlp(nlp.tokenizer.decode(masked_toks), targets=[nlp.tokenizer.decode(dtok1), nlp.tokenizer.decode(dtok2)])
  return res[0]['token'] == dtok1


def mlm_accuracy(sentpairs):
  res = [fill_one(s1, s2) for (s1, s2) in sentpairs]
  return sum(res) / len(sentpairs)

for task_name, sents in sent_pairs.items():
  print(task_name, mlm_accuracy(sents))
