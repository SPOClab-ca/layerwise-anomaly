import sklearn.mixture
import numpy as np

import src.sent_encoder


class AnomalyModel:
  """Model that uses GMM on embeddings generated by BERT for finding syntactic
  or semantic anomalies.
  """

  def __init__(self, train_sentences, model_name='roberta-base',
      model_type='gmm', n_components=1, covariance_type='full',
      svm_kernel='rbf'):
    self.enc = src.sent_encoder.SentEncoder(model_name=model_name)
    self.gmms = []

    _, all_vecs = self.enc.contextual_token_vecs(train_sentences)
    for layer in range(13):
      sent_vecs = np.vstack([vs[:,layer,:] for vs in all_vecs])

      if model_type == 'gmm':
        gmm = sklearn.mixture.GaussianMixture(n_components=n_components, covariance_type=covariance_type)
      elif model_type == 'svm':
        gmm = sklearn.svm.OneClassSVM(kernel=svm_kernel)

      gmm.fit(sent_vecs)
      self.gmms.append(gmm)


  def gmm_score(self, sentences):
    """Returns (all_tokens, all_scores), where
    all_tokens is List[List[token]]
    all_scores is List[np.array(num layers, |S|)]
    """

    all_tokens, all_vecs = self.enc.contextual_token_vecs(sentences)
    all_scores = []

    for sent_ix in range(len(sentences)):
      tokens = all_tokens[sent_ix]
      vecs = all_vecs[sent_ix]
      assert len(tokens) == vecs.shape[0]
      
      layer_scores = []
      for layer in range(13):
        scores = self.gmms[layer].score_samples(vecs[:, layer, :])
        layer_scores.append(scores)

      all_scores.append(np.array(layer_scores))

    return all_tokens, all_scores


  def eval_sent_pairs(self, sentpairs, layer):
    """Evaluate sentence pairs, assuming first pair is correct one.
    Return list of booleans, true if correct one has higher likelihood.
    """
    correct_scores = self.gmm_score([sp[0] for sp in sentpairs])[1]
    correct_scores = [np.sum(sent_scores[layer]) for sent_scores in correct_scores]
    incorrect_scores = self.gmm_score([sp[1] for sp in sentpairs])[1]
    incorrect_scores = [np.sum(sent_scores[layer]) for sent_scores in incorrect_scores]

    return [x > y for (x,y) in zip(correct_scores, incorrect_scores)]
