import os
import pandas as pd


class SentPairGenerator():
  """For now, all sentence pairs are in CSV files"""

  def __init__(self, data_dir='../data'):
    self.data_dir = data_dir

  def get_csv_based_dataset(self, csvname, correct_col, wrong_col):
    """Get two columns of a csv dataset"""
    df = pd.read_csv(os.path.join(self.data_dir, csvname))
    df = df[[correct_col, wrong_col]]
    return [tuple(x) for x in df.to_numpy()]
