import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 2
BATCH_SIZE = 256
MAX_EPOCHS = 400
LR = 5e-4
LR_DECAY_FACTOR = 0.5
LR_DECAY_PATIENCE = 200
LR_MIN = 1e-8
