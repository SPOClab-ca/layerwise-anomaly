import unittest
import src.sent_encoder
import numpy as np

EPS = 1e-9

class TestSentEncoder(unittest.TestCase):

  def test_roberta(self):
    encoder = src.sent_encoder.SentEncoder(model_name='roberta-base')
    sents = ['Good morning', 'You are drunk']
    all_tokens, all_vecs = encoder.contextual_token_vecs(sents)

    assert len(all_vecs) == 2
    assert all_vecs[0].shape == (2, 13, 768)
    assert all_vecs[1].shape == (3, 13, 768)


  def test_xlnet(self):
    encoder = src.sent_encoder.SentEncoder(model_name='xlnet-base-cased')
    sents = ['Good morning', 'You are drunk']
    all_tokens, all_vecs = encoder.contextual_token_vecs(sents)

    assert len(all_vecs) == 2
    assert all_vecs[0].shape == (2, 13, 768)
    assert all_vecs[1].shape == (3, 13, 768)


  def test_roberta_large(self):
    encoder = src.sent_encoder.SentEncoder(model_name='roberta-large')
    sents = ['Good morning', 'You are drunk']
    all_tokens, all_vecs = encoder.contextual_token_vecs(sents)

    assert len(all_vecs) == 2
    assert all_vecs[0].shape == (2, 25, 1024)
    assert all_vecs[1].shape == (3, 25, 1024)
