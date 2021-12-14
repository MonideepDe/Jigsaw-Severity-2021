import torch

class Config:
    data_dir = 'data/'
    model_dir = 'models/'
    input_files = {
        'comments': 'comments_to_score.csv',
        'train': 'toxic.csv',
        'valid': 'validation_data.csv'
    }
    n_folds = 5
    
    toxic_threshold = 0.5
    non_toxic_threshold = 0.85

    model_name = "roberta-base"
    max_len = 128
    bert_output_logits = 768
    dropout = 0.2
    save_name = f'jigsaw_{model_name}.pt'

    seed = 123

    # Wish apple GPU could have been utilized.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_test_split = 0.8 # Not used if KFolds are being used
    batch_size = 64
    epochs = 10
    data_loader_workers = 2
    learning_rate = 2e-5
    max_lr = 5e-5             #OneCycleLR
    pct_start = 0.1           # OneCycleLR
    anneal_strategy = 'cos'   # OneCycleLR
    div_factor= 1e3           # OneCycleLR
    final_div_factor = 1e3    # OneCycleLR