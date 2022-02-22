import torch
from config import Config

def save_model(epoch, model_state_dict, optim_state_dict):
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optim_state_dict': optim_state_dict
        },
        Config.model_name + Config.save_name
    )
    print(f'Saved model to {Config.save_name}')

def load_model(model, optimizer):
    checkpoint = torch.load(Config.save_name)
    model = model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch, model, optimizer