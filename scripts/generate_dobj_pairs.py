"""
Script to generate pairs of sentences in direct / indirect object alternation, using
CheckList and RoBERTa.
"""
import checklist
from checklist.editor import Editor
import pickle

VERBS = ['gave', 'sent', 'mailed', 'brought', 'showed', 'sold']

editor = Editor()

sentences = set()
for vb in VERBS:
  # Only use the first 3 masked words. The second sentence seems to make it generate
  # sentences that are equally likely in both syntactic positions.
  ret = editor.suggest(
    f'The {{mask}} {vb} the {{mask}} a {{mask}}. The {{mask}} {vb} a {{mask}} to the {{mask}}.',
  )

  for t in ret:
    subj = t[0]
    iobj = t[1]
    dobj = t[2]
    if subj == iobj or iobj == dobj or subj == dobj:
      continue
    sent1 = f'The {subj} {vb} the {iobj} a {dobj}.'
    sent2 = f'The {subj} {vb} a {dobj} to the {iobj}.'
    sentences.add((sent1, sent2))

for sent1, sent2 in sentences:
  print(sent1, sent2)

with open('sents.pkl', 'wb') as f:
  pickle.dump(sentences, f)
