import unittest
import src.sent_encoder
import numpy as np

class TestSentEncoder(unittest.TestCase):
  def setUp(self):
    self.enc = src.sent_encoder.SentEncoder()

  def test_diff_encoder(self):
    sent_pairs = [('I am a cat', 'I am a dog'), ('Life is good', 'Life is good')]
    vecs = self.enc.evaluate_contextual_diff(sent_pairs)

    assert vecs.shape == (2, 768)
    assert not np.all(vecs[0] == 0)
    assert np.all(vecs[1] == 0)
