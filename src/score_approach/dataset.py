import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from config import Config
import pandas as pd


class BertDataset(Dataset):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
        self.data = self.read_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data['text'][idx]
        target = self.data['score'][idx] if 'score' in self.data.columns else -1
        tokenized = self.tokenizer(text,
                                   add_special_tokens=True,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=Config.max_len,
                                   return_token_type_ids=True)
        if 'score' in self.data.columns:
            return {
                'input_ids': torch.tensor(tokenized['input_ids'], dtype=torch.long),
                'token_type_ids': torch.tensor(tokenized['token_type_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(tokenized['attention_mask'], dtype=torch.long),
                'target': torch.tensor(target, dtype=torch.float)
            }
        else:
            return {
                'input_ids': torch.tensor(tokenized['input_ids'], dtype=torch.long),
                'token_type_ids': torch.tensor(tokenized['token_type_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(tokenized['attention_mask'], dtype=torch.long),
            }

    def read_data(self):
        df = pd.read_csv(Config.data_dir + Config.input_files['train'])
        data = pd.DataFrame(columns=['text', 'score'])

        data = data.append(df[df.score > Config.toxic_threshold])
        toxic_len = len(data)

        data = data.append(df[df.score == 0].sample(
            n=int(toxic_len * Config.non_toxic_threshold), random_state=Config.seed))
        data.reset_index(inplace=True, drop=True)
        return data


if __name__ == "__main__":
    dataset = BertDataset()
    print(dataset[0])