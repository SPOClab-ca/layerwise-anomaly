"""
Evaluate the accuracy of MLM/GMM on all tasks using specified LM (Table 2 in paper).

Example usage:
PYTHONPATH=. time python scripts/run_accuracy.py \
  --model_name=roberta-base \
  --anomaly_model=gmm
"""
import argparse
from collections import defaultdict
import pickle
import random
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM, XLNetLMHeadModel

import src.sentpair_generator
import src.anomaly_model


BEST_LAYER_FOR_GMM = {
  'roberta-base': 11,
  'bert-base-uncased': 9,
  'xlnet-base-cased': 6,
}


def get_common_sentences():
  """Get filtered sentences where the differing token is in-vocab for all the
  models, for fair comparison.
  """

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

  return sent_pairs


def get_masked_sequence(tokenizer, sent1, sent2):
  """Given a sentence pair that differ on exactly one token, return the token sequence
  with [MASK] token, index of mask token, and the two different tokens.
  """
  toks1 = tokenizer(sent1)['input_ids']
  toks2 = tokenizer(sent2)['input_ids']

  seq_with_mask = []
  masked_ix = None
  dtok1 = None
  dtok2 = None
  for ix in range(len(toks1)):
    if toks1[ix] != toks2[ix]:
      # Sequence is different here, so replace with [MASK].
      seq_with_mask.append(tokenizer.mask_token_id)
      masked_ix = ix
      dtok1 = toks1[ix]
      dtok2 = toks2[ix]
    else:
      seq_with_mask.append(toks1[ix])

  return seq_with_mask, masked_ix, dtok1, dtok2


def run_mlm_mask_accuracy(model_name):
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  # These do the same thing, except XLNet is a less popular model so it's not supported by
  # all AutoModel variants.
  if 'xlnet' in model_name:
    model = XLNetLMHeadModel.from_pretrained(model_name)
  else:
    model = AutoModelForMaskedLM.from_pretrained(model_name)

  # Make binary choice for a single sentence pair
  def mlm_sentence_pair(sent1, sent2):
    masked_toks, masked_ix, dtok1, dtok2 = get_masked_sequence(tokenizer, sent1, sent2)
    logit1 = model(torch.tensor([masked_toks])).logits[0, masked_ix, dtok1]
    logit2 = model(torch.tensor([masked_toks])).logits[0, masked_ix, dtok2]
    return bool(logit1 > logit2)

  sent_pairs = get_common_sentences()
  for task_name, sents in sent_pairs.items():
    res = [mlm_sentence_pair(s1, s2) for (s1, s2) in sents]
    acc = sum(res) / len(sents)
    print(task_name, acc)


def run_gmm_accuracy(model_name, layer):
  with open('data/bnc.pkl', 'rb') as f:
    bnc_sentences = pickle.load(f)

  random.seed(12345)
  bnc_sentences = random.sample(bnc_sentences, 1000)

  model = src.anomaly_model.AnomalyModel(bnc_sentences, model_name=model_name)

  sent_pairs = get_common_sentences()

  # Manually add ROLE-88 which was filtered out
  sentgen = src.sentpair_generator.SentPairGenerator(data_dir='./data')
  sent_pairs['Chow et al.'] = sentgen.get_hand_selected()['Chow et al.'].sent_pairs

  for task_name, sents in sent_pairs.items():
    res = model.eval_sent_pairs(sents, layer)
    acc = len([x for x in res if x > 0]) / len(res)
    print(task_name, acc)


def main():
  parser = argparse.ArgumentParser()

  # Choices: roberta-base, bert-base-uncased, xlnet-base-cased
  parser.add_argument('--model_name', type=str, default='roberta-base')

  # Choices: gmm, mlm-mask
  parser.add_argument('--anomaly_model', type=str, default='mlm-mask')

  args = parser.parse_args()

  if args.anomaly_model == 'mlm-mask':
    run_mlm_mask_accuracy(args.model_name)
  elif args.anomaly_model == 'gmm':
    run_gmm_accuracy(args.model_name, BEST_LAYER_FOR_GMM[args.model_name])
  else:
    raise AssertionError("Not supported")


main()
