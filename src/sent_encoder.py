from transformers import AutoTokenizer, AutoModel
import torch

class SentEncoder:
  def __init__(self, model_name='roberta-base'):
    self.model_name = model_name
    self.auto_tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.auto_model = AutoModel.from_pretrained(model_name)

  def evaluate_contextual_diff(self, pair):
    source, target = pair[0], pair[1]
    src_ids = torch.tensor(self.auto_tokenizer.encode(source)).unsqueeze(0)
    src_vec = self.auto_model(src_ids)[0].mean(dim=1)[0]  # (768,) torch.tensor
    
    tgt_ids = torch.tensor(self.auto_tokenizer.encode(target)).unsqueeze(0)
    tgt_vec = self.auto_model(tgt_ids)[0].mean(dim=1)[0]
    
    d_emb = len(src_vec)  # 768
    diff = src_vec - tgt_vec
    return diff # 768-dimensional torch.tensor
