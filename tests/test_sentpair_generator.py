import unittest
import src.sentpair_generator

class TestSentpairGenerator(unittest.TestCase):

  def test_get_csv_dataset(self):
    gen = src.sentpair_generator.SentPairGenerator(data_dir='./data')
    datasets = gen.get_all_datasets()

    assert 'Pylkkanen' in datasets
    assert datasets['Pylkkanen'].category == 'Selectional'
    assert len(datasets['Pylkkanen'].sent_pairs) == 70
    assert datasets['Pylkkanen'].sent_pairs[0][0] == "the pilot flew the airplane after the intense class"
    assert datasets['Pylkkanen'].sent_pairs[0][1] == "the pilot amazed the airplane after the intense class"


  def test_get_blimp_dataset(self):
    gen = src.sentpair_generator.SentPairGenerator(data_dir='./data')
    datasets = gen.get_all_datasets()

    assert 'BLiMP-SubjVerb1' in datasets
    assert datasets['BLiMP-SubjVerb1'].category == 'Morphosyntax'
    assert len(datasets['BLiMP-SubjVerb1'].sent_pairs) == 1000
    assert datasets['BLiMP-SubjVerb1'].sent_pairs[0][0] == "Paula references Robert."
    assert datasets['BLiMP-SubjVerb1'].sent_pairs[0][1] == "Paula reference Robert."


  def test_get_blimp_all(self):
    gen = src.sentpair_generator.SentPairGenerator(data_dir='./data')
    datasets = gen.get_blimp_all()

    assert len(datasets) == 67
    assert list(datasets.keys())[0] == 'anaphor_gender_agreement'
    assert datasets['anaphor_gender_agreement'].category == 'anaphor_agreement'
    assert len(datasets['anaphor_gender_agreement'].sent_pairs) == 1000
