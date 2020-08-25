"""
Script to process XML files from BNC corpus and save data into Python Pickle format
that's faster to read.

Usage:
  python scripts/process_bnc.py --bnc_dir=data/bnc/download/Texts --to=data/bnc/bnc.pkl
"""

import argparse
import nltk.corpus.reader.bnc
import tqdm
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--bnc_dir', type=str)
parser.add_argument('--to_file', type=str)
args = parser.parse_args()
print(args)

bnc_reader = nltk.corpus.reader.bnc.BNCCorpusReader(root=args.bnc_dir, fileids=r'[a-z]{3}/\w*\.xml')

sentences = []
for sent in tqdm.tqdm(bnc_reader.sents(strip_space=False)):
  if len(sent) > 3:
    sentences.append(''.join(sent))

with open(args.to_file, 'wb') as f:
  pickle.dump(sentences, f)

print('Done')
