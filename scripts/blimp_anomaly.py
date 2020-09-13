"""
Run GMM on BLiMP syntactic minimal pairs.

Example usage:
PYTHONPATH=. time python scripts/blimp_anomaly.py \
  --bnc_path=data/bnc.pkl \
  --blimp_path=data/blimp/data/ \
  --out_file=blimp_result.txt \
  --num_gmm_sentences 1000
"""
import argparse
import glob
import jsonlines
import pickle
import random
import src.anomaly_model


parser = argparse.ArgumentParser()
parser.add_argument('--bnc_path', type=str)
parser.add_argument('--blimp_path', type=str)
parser.add_argument('--out_file', type=str)
parser.add_argument('--num_gmm_sentences', type=int)
args = parser.parse_args()
print(args)


print('Loading BNC...')

with open(args.bnc_path, 'rb') as f:
  bnc_sentences = pickle.load(f)

random.seed(12345)
bnc_sentences = random.sample(bnc_sentences, args.num_gmm_sentences)

print('Training GMM...')
model = src.anomaly_model.AnomalyModel(bnc_sentences)


def process_blimp(fname):
  with jsonlines.open(fname) as reader:
    lines = list(reader)

  task_name = fname.split('/')[-1][:-len('.jsonl')]
  print('Processing:', task_name)

  sentpairs = [(l['sentence_good'], l['sentence_bad']) for l in lines]
  
  for layer in range(13):
    result = sum(model.eval_sent_pairs(sentpairs, layer)) / len(sentpairs)
    with open(args.out_file, 'a') as outf:
      print(task_name, layer, result)
      print(task_name, layer, result, file=outf)


for fname in glob.glob(args.blimp_path + '/**.jsonl'):
  process_blimp(fname)
