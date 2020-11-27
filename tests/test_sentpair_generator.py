import unittest
import src.sentpair_generator

class TestSentpairGenerator(unittest.TestCase):
  def test_get_all_datasets(self):
    gen = src.sentpair_generator.SentPairGenerator(data_dir='./data')
    datasets = gen.get_all_datasets()

    assert 'Pylkkanen' in datasets
    assert datasets['Pylkkanen'].category == 'Selectional'
    assert len(datasets['Pylkkanen'].sent_pairs) == 70
    assert datasets['Pylkkanen'].sent_pairs[0][0] == "the pilot flew the airplane after the intense class"
    assert datasets['Pylkkanen'].sent_pairs[0][1] == "the pilot amazed the airplane after the intense class"
