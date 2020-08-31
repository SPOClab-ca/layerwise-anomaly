import unittest
import src.anomaly_model

class TestAnomalyModel(unittest.TestCase):

  def setUp(self):
    train_sentences = ['Good morning', 'You are drunk']
    self.model = src.anomaly_model.AnomalyModel(train_sentences)


  def test_init(self):
    assert len(self.model.gmms) == 13


  def test_gmm_score(self):
    tokens, layer_scores = self.model.gmm_score('I like pigs.')
    assert tokens == ['I', ' like', ' pigs', '.']
    assert layer_scores.shape == (13, 4)
