import torch

CONFIG = dict(
    n_folds = 5,
    
    seed = 2022,
    
    # Model config
    model_name = "xlm-roberta-base",
    max_len = 128,
    bert_output_logits = 768,
    dropout = 0.2,
    extra_dropout = 0.1,
    
    # Train config
    train_test_split = 0.8, # Not used if KFolds are being used
    batch_size = 64,
    epochs = 1,
    data_loader_workers = 2,
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    learning_rate = 2e-5,
    max_lr = 5e-5,             #OneCycleLR
    pct_start = 0.1,           # OneCycleLR
    anneal_strategy = 'cos',   # OneCycleLR
    div_factor= 1e3,           # OneCycleLR
    final_div_factor = 1e3,    # OneCycleLR
    
    # Loss config
    margin = 0.2    # Recommended - to be in sync with margin in data prep / cleaning config

)