import transformers
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import torch

BATCH_SIZE = 32

class SentEncoder:
  def __init__(self, model_name='roberta-base'):
    self.model_name = model_name
    if 'chinese' in model_name:
      self.auto_tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
    else:
      self.auto_tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.auto_model = AutoModel.from_pretrained(model_name).cuda()


  def contextual_token_vecs(self, sents):
    """Returns: (all_tokens, sentence_token_vecs) where:
    all_tokens is a List[List[tokens]], one list for each sentence.
    sentence_token_vecs is List[np.array(sentence length, 13, 768)], one array for each sentence.
    Ignore special tokens like [CLS] and [PAD].
    """
    all_tokens = []
    sentence_token_vecs = []

    for batch_ix in range(0, len(sents), BATCH_SIZE):
      batch_sentences = sents[batch_ix : batch_ix+BATCH_SIZE]

      ids = torch.tensor(self.auto_tokenizer(batch_sentences, padding=True)['input_ids']).cuda()

      with torch.no_grad():
        # (num_layers, batch_size, sent_length, 768)
        vecs = self.auto_model(ids, attention_mask=(ids != 1), output_hidden_states=True)[2]
        vecs = np.array([v.detach().cpu().numpy() for v in vecs])

      for sent_ix in range(ids.shape[0]):
        tokens = []
        token_vecs = []

        for tok_ix in range(ids.shape[1]):
          if ids[sent_ix, tok_ix] not in self.auto_tokenizer.all_special_ids:
            tokens.append(self.auto_tokenizer.decode(int(ids[sent_ix, tok_ix])))
            token_vecs.append(vecs[:, sent_ix, tok_ix, :])

        all_tokens.append(tokens)
        sentence_token_vecs.append(np.array(token_vecs))

    return all_tokens, sentence_token_vecs


  def _mean_without_pad(self, batch_ids, batch_vecs):
    """Must not include [PAD] tokens when averaging token embeddings"""
    positions_not_pad = (batch_ids != self.auto_tokenizer.pad_token_id).to(float).unsqueeze(2)
    batch_vecs = positions_not_pad * batch_vecs
    return batch_vecs.sum(dim=1) / positions_not_pad.sum(dim=1)


  def evaluate_contextual_diff(self, pairs, layer=-2):
    """Get sentence embedding difference between pairs of sentences, for a given layer.
    Sentence embeddings are generated by averaging across contextual word embeddings."""
    result = []
    for batch_ix in range(0, len(pairs), BATCH_SIZE):
      batch_sentences = pairs[batch_ix : batch_ix+BATCH_SIZE]
      src_sentences = [p[0] for p in batch_sentences]
      tgt_sentences = [p[1] for p in batch_sentences]

      src_ids = torch.tensor(self.auto_tokenizer(src_sentences, padding=True)['input_ids']).cuda()
      tgt_ids = torch.tensor(self.auto_tokenizer(tgt_sentences, padding=True)['input_ids']).cuda()

      # Needed to avoid leaking cuda memory
      with torch.no_grad():
        src_vecs = self.auto_model(src_ids, attention_mask=(src_ids != 1), output_hidden_states=True)[2][layer]
        tgt_vecs = self.auto_model(tgt_ids, attention_mask=(tgt_ids != 1), output_hidden_states=True)[2][layer]

      src_sent_vecs = self._mean_without_pad(src_ids, src_vecs)
      tgt_sent_vecs = self._mean_without_pad(tgt_ids, tgt_vecs)
      diff_sent_vecs = src_sent_vecs - tgt_sent_vecs
      result.append(diff_sent_vecs.detach().cpu().numpy())

    return np.vstack(result)


  def get_layer_distance_df(self, sentence_pairs):
    """Get Euclidean distance for list of sentence pairs, for all layers, in a dataframe."""
    distances = []
    for layer in range(13):
      vecs = self.evaluate_contextual_diff(sentence_pairs, layer=layer)

      for ix in range(len(sentence_pairs)):
        dist = np.linalg.norm(vecs[ix])
        distances.append(pd.Series({
          'layer': layer,
          'dist': dist,
          'sent1': sentence_pairs[ix][0],
          'sent2': sentence_pairs[ix][1],
        }))
    return pd.DataFrame(distances)
