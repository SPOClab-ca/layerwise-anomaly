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


  def get_hand_selected(self):
    """A hand-selected set of psycholinguistic pairs and BLiMP pairs"""
    datasets = {}

    datasets['BLiMP (Subject-Verb)'] = SentPairSet(
      category='Morphosyntax',
      sent_pairs=self.load_blimp('blimp/data/regular_plural_subject_verb_agreement_1.jsonl') + \
                 self.load_blimp('blimp/data/regular_plural_subject_verb_agreement_2.jsonl')
    )

    datasets['BLiMP (Det-Noun)'] = SentPairSet(
      category='Morphosyntax',
      sent_pairs=self.load_blimp('blimp/data/determiner_noun_agreement_1.jsonl') + \
                 self.load_blimp('blimp/data/determiner_noun_agreement_2.jsonl')
    )

    datasets['Osterhout and Nicol - Syntactic'] = SentPairSet(
      category='Morphosyntax',
      sent_pairs=self.get_csv_based_dataset('osterhout-nicol.csv', 'original_sentence', 'syntactic_anomaly')
    )

    datasets['BLiMP (Animacy)'] = SentPairSet(
      category='Semantic',
      sent_pairs=self.load_blimp('blimp/data/animate_subject_passive.jsonl') + \
                 self.load_blimp('blimp/data/animate_subject_trans.jsonl')
    )

    datasets['Pylkkanen and McElree'] = SentPairSet(
      category='Semantic',
      sent_pairs=self.get_csv_based_dataset('pylkkanen-mcelree.csv', 'sent_control', 'sent_anomaly')
    )

    datasets['Warren et al. - Selectional'] = SentPairSet(
      category='Semantic',
      sent_pairs=self.get_csv_based_dataset('warren-et-al.csv', 'sent_control', 'sent_violation')
    )

    datasets['Osterhout and Nicol - Semantic'] = SentPairSet(
      category='Semantic',
      sent_pairs=self.get_csv_based_dataset('osterhout-nicol.csv', 'original_sentence', 'semantic_anomaly')
    )

    datasets['Osterhout and Mobley'] = SentPairSet(
      category='Semantic',
      sent_pairs=self.get_csv_based_dataset('osterhout-mobley.csv', 'sent_correct', 'sent_wrong')
    )

    datasets['Warren et al. - Pragmatic'] = SentPairSet(
      category='Commonsense',
      sent_pairs=self.get_csv_based_dataset('warren-et-al.csv', 'sent_control', 'sent_no_violation')
    )
    
    datasets['Federmeier and Kutas'] = SentPairSet(
      category='Commonsense',
      sent_pairs=self.get_csv_based_dataset('federmeier-kutas.csv', 'correct', 'incorrect')
    )

    datasets['Chow et al.'] = SentPairSet(
      category='Commonsense',
      sent_pairs=self.get_csv_based_dataset('chow-et-al.csv', 'correct', 'reversed')
    )

    datasets['Urbach and Kutas'] = SentPairSet(
      category='Commonsense',
      sent_pairs=self.get_csv_based_dataset('urbach-kutas.csv', 'sent_correct', 'sent_wrong')
    )

    return datasets


  def get_blimp_all(self, subtasks=True):
    """Get all BLiMP tasks.
    If subtasks is True, then return 67 tasks, one for each paradigm.
    If subtasks is False, then return 12 tasks, grouping together paradigms by linguistic phenomenon.
    """
    blimp_tasks = pd.read_csv(os.path.join(self.data_dir, 'blimp/raw_results/blimp_full_results_summary.csv'))
    blimp_tasks = blimp_tasks.groupby('UID', sort=False).first().reset_index()

    datasets = {}

    if subtasks:
      for _, task in blimp_tasks.iterrows():
        datasets[task['UID']] = SentPairSet(
          category=task['linguistics_term'],
          sent_pairs=self.load_blimp(f"blimp/data/{task['UID']}.jsonl")
        )

    else:
      grouped_sents = collections.defaultdict(list)
      for _, task in blimp_tasks.iterrows():
        grouped_sents[task['linguistics_term']].extend(self.load_blimp(f"blimp/data/{task['UID']}.jsonl"))

      for task_name, task_sents in grouped_sents.items():
        datasets[task_name] = SentPairSet(
          category=task_name,
          sent_pairs=task_sents
        )

    return datasets
