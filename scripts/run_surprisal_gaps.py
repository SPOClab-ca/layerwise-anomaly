"""
Train Gaussian model and calculate surprisal gaps for task (Figure 4 in paper).

Example usage:
PYTHONPATH=. time python scripts/run_surprisal_gaps.py \
  --bnc_path=data/bnc.pkl \
  --out=surprisal_gaps
"""
import argparse
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import src.anomaly_model
import src.sentpair_generator


parser = argparse.ArgumentParser()

# Prefix for output files
parser.add_argument('--out', type=str, default='default')

# Paths to data files
parser.add_argument('--bnc_path', type=str, default='data/bnc.pkl')

# Model hyperparameters
parser.add_argument('--model_name', type=str, default='roberta-base')
parser.add_argument('--model_type', type=str, default='gmm')
parser.add_argument('--num_gmm_sentences', type=int, default=1000)
parser.add_argument('--n_components', type=int, default=1)
parser.add_argument('--covariance_type', type=str, default='full')
parser.add_argument('--svm_kernel', type=str, default='rbf')

# Evaluation parameters
parser.add_argument('--eval_dataset', type=str, default='hand-selected')
parser.add_argument('--max_eval_sents', type=int, default=100)

args = parser.parse_args()

# Log args to file
with open(args.out + '.log', 'w') as outf:
  print(args)
  print(args, file=outf)


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

sentgen = src.sentpair_generator.SentPairGenerator(data_dir='./data')

def process_sentpair_dataset(taskname, category, sent_pairs):
  print(f"Processing: {category} - {taskname}")
  if len(sent_pairs) > args.max_eval_sents:
    sent_pairs = random.sample(sent_pairs, args.max_eval_sents)
  
  scores = []
  for layer in range(model.num_model_layers):
    results = model.eval_sent_pairs(sent_pairs, layer)
    scores.extend([{'category': category, 'taskname': taskname, 'layer': layer, 'score': r} for r in results])
  scores = pd.DataFrame(scores)
  return scores


# Pick the dataset
if args.eval_dataset == 'hand-selected':
  eval_dataset = sentgen.get_hand_selected()
elif args.eval_dataset == 'blimp12':
  eval_dataset = sentgen.get_blimp_all(subtasks=False)
elif args.eval_dataset == 'blimp67':
  eval_dataset = sentgen.get_blimp_all(subtasks=True)
else:
  assert(False)


# Process all datasets and calculate surprisal gaps
all_scores = []
for taskname, sent_pair_set in eval_dataset.items():
  task_scores = process_sentpair_dataset(taskname, sent_pair_set.category, sent_pair_set.sent_pairs)
  all_scores.append(task_scores)
all_scores = pd.concat(all_scores)

surprisal_gaps = all_scores.groupby(['category', 'taskname', 'layer'], sort=False).score \
  .aggregate(lambda x: np.mean(x) / np.std(x)).reset_index()

surprisal_gaps['task'] = surprisal_gaps.apply(lambda r: f"{r['category']} - {r['taskname']}", axis=1)
surprisal_gaps = surprisal_gaps[['task', 'layer', 'score']]
surprisal_gaps.to_csv(f'{args.out}.csv', index=False)


# Save plots
g = sns.FacetGrid(surprisal_gaps, row="task", height=2, aspect=4.5)
g.map_dataframe(sns.barplot, x="layer", y="score")
g.set_axis_labels("", "Surprisal Gap")
g.set_titles(row_template="{row_name}")
g.set(ylim=(-1.5, 3))
plt.savefig(f'{args.out}.png')
plt.savefig(f'{args.out}.pdf')
