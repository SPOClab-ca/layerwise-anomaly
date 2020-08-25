import unittest
import src.sent_encoder
import numpy as np

EPS = 1e-9

class TestSentEncoder(unittest.TestCase):
  def setUp(self):
    self.enc = src.sent_encoder.SentEncoder()


  def test_diff_encoder(self):
    sent_pairs = [('I am a cat', 'I am not a cat'), ('Life is good', 'Life is good')]
    vecs = self.enc.evaluate_contextual_diff(sent_pairs)

    assert vecs.shape == (2, 768)
    assert np.sum(vecs[0]**2) > EPS
    assert np.sum(vecs[1]**2) < EPS


  def test_gen_contextual(self):
    sents = ['Good morning']
    vecs = self.enc.contextual_token_vecs(sents)
    assert vecs.shape == (2, 768)

    sents = ['Good morning', 'You are drunk']
    vecs = self.enc.contextual_token_vecs(sents)
    assert vecs.shape == (5, 768)
