from torch.serialization import save
from torch.utils.data import DataLoader, SubsetRandomSampler
from config import Config
import numpy as np
from model import BertModel
from dataset import BertDataset
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from engine import train, evaluate
from utils import save_model

def get_data_loaders(dataset, num_workers = Config.data_loader_workers,train_indices=None, test_indices=None):
    if ((train_indices == None) ^ (test_indices == None)):
        # Both train_indices and test_indices must be same, otherwise there is an issue
        # As, if we get indices we will get for both (from StratifiedKFold or likes of it)
        raise Exception("Must provide indices for both testing and training or dont provide indices for either")
    
    if not train_indices:
        dataset_len = len(dataset)
        indices = list(range(dataset_len))
        np.random.shuffle(indices)
        split = int(np.floor(Config.train_test_split * dataset_len))
        train_indices, test_indices = indices[:split], indices[split:]
        
    # For every batch the data should be randomized
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    train_loader = DataLoader(
        dataset=dataset,
        batch_size = Config.batch_size,
        sampler = train_sampler,
        num_workers = num_workers
    )
    test_loader = DataLoader(
        dataset=dataset,
        batch_size = Config.batch_size,
        sampler = test_sampler,
        num_workers = num_workers
    )
    
    return train_loader, test_loader

def run_training():
    dataset = BertDataset()
    train_dataloader, test_dataloader = get_data_loaders(dataset)

    model = BertModel()

    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.max_lr,
        steps_per_epoch=int((Config.train_test_split * len(dataset)) / Config.batch_size) + 1,
        epochs=Config.epochs,
        pct_start=Config.pct_start,
        anneal_strategy=Config.anneal_strategy,
        div_factor=Config.div_factor,
        final_div_factor=Config.final_div_factor
    )
    
    model.to(Config.device)

    for epoch in range(Config.epochs):
        train_loss = train(model, train_dataloader, optimizer, epoch)
        valid_loss, _ = evaluate(model, test_dataloader, epoch)
    
    return epoch, model, optimizer

if __name__ == "__main__":
    epoch, model, optimizer = run_training()
    save_model(epoch, model.state_dict(), optimizer.state_dict())