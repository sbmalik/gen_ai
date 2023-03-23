from dataclasses import dataclass
import os
from dataloader import get_default_device

@dataclass
class BaseConfig:
    DEVICE = get_default_device()
    DATASET = "Flowers" #  "MNIST", "Cifar-10", "Cifar-100", "Flowers"
    
    # For logging inferece images and saving checkpoints.
    root_log_dir = os.path.join("logdir", "Inference")
    root_checkpoint_dir = os.path.join("logdir", "checkpoints")

    # Current log and checkpoint directory.
    log_dir = "version_0"
    checkpoint_dir = "version_0"

@dataclass
class TrainingConfig:
    TIMESTEPS = 1000 # Define number of diffusion timesteps
    IMG_SHAPE = (1, 32, 32) if BaseConfig.DATASET == "MNIST" else (3, 32, 32) 
    NUM_EPOCHS = 800
    BATCH_SIZE = 32
    LR = 2e-4
    NUM_WORKERS = 2

@dataclass
class ModelConfig:
    BASE_CH = 64  # 64, 128, 256, 256
    BASE_CH_MULT = (1, 2, 4, 4) # 32, 16, 8, 8 
    APPLY_ATTENTION = (False, True, True, False)
    DROPOUT_RATE = 0.1
    TIME_EMB_MULT = 4 # 128