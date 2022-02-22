import torch.nn as nn
from transformers import AutoModel

from config import Config

class BertModel(nn.Module):
    def __init__(self):
        super(BertModel, self).__init__()
        self.bert = AutoModel.from_pretrained(Config.model_name, return_dict=False)
        self.layer_norm = nn.LayerNorm(normalized_shape=Config.bert_output_logits)
        self.dropout = nn.Dropout(Config.dropout)
        self.dense = nn.Sequential(
            nn.Linear(Config.bert_output_logits, 128),
            nn.SiLU(),
            nn.Dropout(Config.dropout),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(Config.dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, input_ids, token_type_ids, attention_mask):
        _, pooler_out = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask) # ignoring hidden state
        out = self.layer_norm(pooler_out)
        out = self.dropout(out)
        preds = self.dense(out)
        preds = preds.squeeze()
        return preds