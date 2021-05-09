"""
Run GMM on BLiMP syntactic minimal pairs (Figure 2 and Appendix A in paper).

Example usage:
PYTHONPATH=. time python scripts/blimp_anomaly.py \
  --bnc_path=data/bnc.pkl \
  --blimp_path=data/blimp/data/ \
  --out=blimp_result
"""
import argparse
import glob
import jsonlines
import pickle
import random
import src.anomaly_model


parser = argparse.ArgumentParser()

# Prefix for output files
parser.add_argument('--out', type=str, default='default')

# Paths to data files
parser.add_argument('--bnc_path', type=str)
parser.add_argument('--blimp_path', type=str)

# Model hyperparameters
parser.add_argument('--model_name', type=str, default='roberta-base')
parser.add_argument('--model_type', type=str, default='gmm')
parser.add_argument('--num_gmm_sentences', type=int, default=1000)
parser.add_argument('--n_components', type=int, default=1)
parser.add_argument('--covariance_type', type=str, default='full')
parser.add_argument('--svm_kernel', type=str, default='rbf')

# Evaluation hyperparameters
parser.add_argument('--layer', type=str, default='all')

args = parser.parse_args()

# Log args to file
with open(args.out + '.log', 'w') as outf:
  print(args)
  print(args, file=outf)


if args.layer == 'all':
  eval_layers = range(13)
else:
  eval_layers = [int(args.layer)]


print('Loading BNC...')

with open(args.bnc_path, 'rb') as f:
  bnc_sentences = pickle.load(f)

random.seed(12345)
bnc_sentences = random.sample(bnc_sentences, args.num_gmm_sentences)

print('Training anomaly model...')
model = src.anomaly_model.AnomalyModel(
  bnc_sentences,
  model_name=args.model_name,
  model_type=args.model_type,
  n_components=args.n_components,
  covariance_type=args.covariance_type,
  svm_kernel=args.svm_kernel,
)

def process_blimp(fname):
  with jsonlines.open(fname) as reader:
    lines = list(reader)

  task_name = fname.split('/')[-1][:-len('.jsonl')]
  print('Processing:', task_name)

  sentpairs = [(l['sentence_good'], l['sentence_bad']) for l in lines]
  
  for layer in eval_layers:
    result = sum([x > 0 for x in model.eval_sent_pairs(sentpairs, layer)]) / len(sentpairs)
    with open(args.out + '.log', 'a') as outf:
      print(f"{task_name},{layer},{result}")
      print(f"{task_name},{layer},{result}", file=outf)


for fname in glob.glob(args.blimp_path + '/**.jsonl'):
  process_blimp(fname)
