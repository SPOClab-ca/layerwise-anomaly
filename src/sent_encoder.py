from transformers import AutoTokenizer, AutoModel
import torch

class SentEncoder:
  def __init__(self, model_name='roberta-base'):
    self.model_name = model_name
    self.auto_tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.auto_model = AutoModel.from_pretrained(model_name).cuda()

  def evaluate_contextual_diff(self, pair, layer=-1):
    """Get sentence embedding difference between pair of sentences, for a given layer.
    Sentence embeddings are generated by averaging across contextual word embeddings."""
    source, target = pair[0], pair[1]
    src_ids = torch.tensor(self.auto_tokenizer.encode(source)).unsqueeze(0).cuda()
    src_vec = self.auto_model(src_ids, output_hidden_states=True)[2][layer].mean(dim=1)[0]  # (768,) torch.tensor
    
    tgt_ids = torch.tensor(self.auto_tokenizer.encode(target)).unsqueeze(0).cuda()
    tgt_vec = self.auto_model(tgt_ids, output_hidden_states=True)[2][layer].mean(dim=1)[0]  # (768,) torch.tensor
    
    d_emb = len(src_vec)  # 768
    diff = src_vec - tgt_vec
    return diff # 768-dimensional torch.tensor
