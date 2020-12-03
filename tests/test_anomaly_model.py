import unittest
import src.anomaly_model

class TestAnomalyModel(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    train_sentences = ['Good morning', 'You are drunk']
    cls.model = src.anomaly_model.AnomalyModel(train_sentences)


  def test_init(self):
    assert len(self.model.gmms) == 13


  def test_gmm_score(self):
    all_tokens, all_scores = self.model.gmm_score(['I like pigs.', 'My pig likes to eat.'])

    assert len(all_tokens) == 2
    assert len(all_scores) == 2

    assert all_tokens[0] == ['I', ' like', ' pigs']
    assert all_scores[0].shape == (13, 3)

    assert all_tokens[1] == ['My', ' pig', ' likes', ' to', ' eat']
    assert all_scores[1].shape == (13, 5)


  def test_eval_sent_pairs(self):
    sents = [('Good morning', 'Good afternoon'), ('Good afternoon', 'Good morning'), ('pig', 'pig')]
    results = self.model.eval_sent_pairs(sents, -2)
    assert len(results) == 3
    assert results[0] > 0
    assert results[1] < 0
    assert results[2] == 0
