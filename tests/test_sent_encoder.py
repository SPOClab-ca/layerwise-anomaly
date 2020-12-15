import unittest
import src.sent_encoder
import numpy as np

EPS = 1e-9

class TestSentEncoder(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.enc = src.sent_encoder.SentEncoder(model_name='roberta-base')


  def test_diff_encoder(self):
    sent_pairs = [('I am a cat', 'I am not a cat'), ('Life is good', 'Life is good')]
    vecs = self.enc.evaluate_contextual_diff(sent_pairs)

    assert vecs.shape == (2, 768)
    assert np.sum(vecs[0]**2) > EPS
    assert np.sum(vecs[1]**2) < EPS


  def test_gen_contextual(self):
    sents = ['Good morning', 'You are drunk']
    all_tokens, all_vecs = self.enc.contextual_token_vecs(sents)

    assert len(all_vecs) == 2
    assert all_vecs[0].shape == (2, 13, 768)
    assert all_vecs[1].shape == (3, 13, 768)
    assert all_tokens == [['Good', ' morning'], ['You', ' are', ' drunk']]


  def test_xlnet(self):
    xlnet_enc = src.sent_encoder.SentEncoder(model_name='xlnet-base-cased')
    sents = ['Good morning', 'You are drunk']
    all_tokens, all_vecs = xlnet_enc.contextual_token_vecs(sents)

    assert len(all_vecs) == 2
    assert all_vecs[0].shape == (2, 13, 768)
    assert all_vecs[1].shape == (3, 13, 768)
