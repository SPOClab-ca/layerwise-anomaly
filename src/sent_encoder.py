import transformers
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import string

BATCH_SIZE = 32
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
class SentEncoder:
  def __init__(self, model_name='roberta-base'):
    self.model_name = model_name
    self.auto_tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.auto_model = AutoModel.from_pretrained(model_name).to(device)
    self.pad_id = self.auto_tokenizer.pad_token_id


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

      ids = torch.tensor(self.auto_tokenizer(batch_sentences, padding=True)['input_ids']).to(device)

      with torch.no_grad():
        # (num_layers, batch_size, sent_length, 768)
        vecs = self.auto_model(
          ids,
          attention_mask=(ids != self.pad_id).float(),
          output_hidden_states=True)[2]
        vecs = np.array([v.detach().cpu().numpy() for v in vecs])

      for sent_ix in range(ids.shape[0]):
        tokens = []
        token_vecs = []

        for tok_ix in range(ids.shape[1]):
          if ids[sent_ix, tok_ix] not in self.auto_tokenizer.all_special_ids:
            cur_tok = self.auto_tokenizer.decode(int(ids[sent_ix, tok_ix]))
            # Exclude tokens that consist entirely of punctuation
            if cur_tok not in string.punctuation:
              tokens.append(cur_tok)
              token_vecs.append(vecs[:, sent_ix, tok_ix, :])

        all_tokens.append(tokens)
        sentence_token_vecs.append(np.array(token_vecs))

    return all_tokens, sentence_token_vecs
