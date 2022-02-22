import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from config import CONFIG
import pandas as pd

class BertDataset(Dataset):
    def __init__(self, df):
        self.data = df
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        less_toxic_text = self.data['less_toxic'][idx]
        more_toxic_text = self.data['more_toxic'][idx]
        
        print(f"less_toxic_text: {less_toxic_text}")
        print(f"more_toxic_text: {more_toxic_text}")
        
        #target = self.data['score'][idx] if 'score' in self.data.columns else -1
        target = 1 # For MarginRankingLoss - Since we know less toxic must have less score than more toxic 
        
        less_toxic_tokenized = self.tokenizer(less_toxic_text, 
                      add_special_tokens=True, 
                      truncation=True,
                      padding='max_length',
                      max_length=CONFIG['max_len'],
                      return_token_type_ids=True)
        
        print(f'Less toxic tokenized: {less_toxic_tokenized}')
        
        more_toxic_tokenized = self.tokenizer(more_toxic_text, 
                      add_special_tokens=True, 
                      truncation=True,
                      padding='max_length',
                      max_length=CONFIG['max_len'],
                      return_token_type_ids=True)
        
        item = {
            'less_input_ids': torch.tensor(less_toxic_tokenized['input_ids'], dtype=torch.long),
            'less_token_type_ids': torch.tensor(less_toxic_tokenized['token_type_ids'], dtype=torch.long),
            'less_attention_mask': torch.tensor(less_toxic_tokenized['attention_mask'], dtype=torch.long),
            
            'more_input_ids': torch.tensor(more_toxic_tokenized['input_ids'], dtype=torch.long),
            'more_token_type_ids': torch.tensor(more_toxic_tokenized['token_type_ids'], dtype=torch.long),
            'more_attention_mask': torch.tensor(more_toxic_tokenized['attention_mask'], dtype=torch.long),
        }
        
        if 'score' in self.data.columns:
            item.update({'target': torch.tensor(target, dtype=torch.float)})
        
        return item

if __name__ == "__main__":
    df = pd.read_csv("data/compare_approach/jigsaw_compare_df.csv")
    dataset = BertDataset(df)
    print(dataset[0])