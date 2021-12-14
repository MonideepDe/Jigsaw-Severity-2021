import torch
import torch.nn as nn
from tqdm import tqdm

from config import Config


def train(model, data_loader, optimizer, epoch):
    model.train()
    final_loss = 0
    batch = 1
    
    tk = tqdm(data_loader, total=len(data_loader), position=0, leave=True)

    for data in tk:
        input_ids = data['input_ids'].to(Config.device)
        token_type_ids = data['token_type_ids'].to(Config.device)
        attention_mask = data['attention_mask'].to(Config.device)
        target = data['target'].to(Config.device)
        
        optimizer.zero_grad()
        pred = model(
            input_ids = input_ids, 
            token_type_ids = token_type_ids, 
            attention_mask = attention_mask
        )
        loss = nn.MSELoss()(pred, target)
        loss.backward()
        optimizer.step()
        final_loss += loss.item()
        tk.set_description(f"Epoch {epoch}; Loss: {loss.item():.4f}; Final Loss: {final_loss / batch:.4f}")
        batch += 1
    final_loss = final_loss / len(data_loader)
    return final_loss

def evaluate(model, data_loader, epoch):
    model.eval()
    final_loss = 0
    
    with torch.no_grad():
        tk = tqdm(data_loader, total=len(data_loader), position=0, leave=True)
        batch = 1
        all_preds = []
        for data in tk:
            input_ids = data['input_ids'].to(Config.device)
            token_type_ids = data['token_type_ids'].to(Config.device)
            attention_mask = data['attention_mask'].to(Config.device)
            target = data['target'].to(Config.device)

            pred = model(
                input_ids = input_ids, 
                token_type_ids = token_type_ids, 
                attention_mask = attention_mask
            )
            loss = nn.MSELoss()(pred, target)
            final_loss += loss.item()
            tk.set_description(f"Validation:: Epoch {epoch}; Loss: {loss.item():.4f}; Final Loss: {final_loss / batch:.4f}")
            batch += 1
            all_preds.append(pred)
        final_loss = final_loss / len(data_loader)
        return all_preds, final_loss
