import os
import collections
import pandas as pd
import jsonlines


SentPairSet = collections.namedtuple('SentPairSet', 'category sent_pairs')


class SentPairGenerator():
  """For now, all sentence pairs are in CSV files"""

  def __init__(self, data_dir='../data'):
    self.data_dir = data_dir


  def get_csv_based_dataset(self, csvname, correct_col, wrong_col):
    """Get two columns of a csv dataset"""
    df = pd.read_csv(os.path.join(self.data_dir, csvname))
    df = df[[correct_col, wrong_col]]
    return [tuple(x) for x in df.to_numpy()]


  def load_blimp(self, fname):
    fname = os.path.join(self.data_dir, fname)
    with jsonlines.open(fname) as reader:
      lines = list(reader)

    return [(l['sentence_good'], l['sentence_bad']) for l in lines]


  def get_all_datasets(self):
    datasets = {}

    datasets['BLiMP-SubjVerb1'] = SentPairSet(
      category='Morphosyntax',
      sent_pairs=self.load_blimp('blimp/data/regular_plural_subject_verb_agreement_1.jsonl')
    )

    datasets['BLiMP-SubjVerb2'] = SentPairSet(
      category='Morphosyntax',
      sent_pairs=self.load_blimp('blimp/data/regular_plural_subject_verb_agreement_2.jsonl')
    )

    datasets['BLiMP-DetNoun1'] = SentPairSet(
      category='Morphosyntax',
      sent_pairs=self.load_blimp('blimp/data/determiner_noun_agreement_1.jsonl')
    )

    datasets['BLiMP-DetNoun2'] = SentPairSet(
      category='Morphosyntax',
      sent_pairs=self.load_blimp('blimp/data/determiner_noun_agreement_2.jsonl')
    )

    datasets['BLiMP-AnimacyPassive'] = SentPairSet(
      category='Selectional',
      sent_pairs=self.load_blimp('blimp/data/animate_subject_passive.jsonl')
    )

    datasets['BLiMP-AnimacyTransitive'] = SentPairSet(
      category='Selectional',
      sent_pairs=self.load_blimp('blimp/data/animate_subject_trans.jsonl')
    )

    datasets['Pylkkanen'] = SentPairSet(
      category='Selectional',
      sent_pairs=self.get_csv_based_dataset('pylkkanen.csv', 'sent_control', 'sent_anomaly')
    )

    datasets['Warren-Selectional'] = SentPairSet(
      category='Selectional',
      sent_pairs=self.get_csv_based_dataset('warren.csv', 'sent_control', 'sent_violation')
    )

    datasets['Warren-Pragmatic'] = SentPairSet(
      category='Pragmatic',
      sent_pairs=self.get_csv_based_dataset('warren.csv', 'sent_control', 'sent_no_violation')
    )
    
    datasets['CPRAG-34'] = SentPairSet(
      category='Pragmatic',
      sent_pairs=self.get_csv_based_dataset('cprag34.csv', 'correct', 'incorrect')
    )

    datasets['ROLE-88'] = SentPairSet(
      category='Pragmatic',
      sent_pairs=self.get_csv_based_dataset('role88.csv', 'correct', 'reversed')
    )

    return datasets
