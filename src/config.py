import torch

class Config:
    # Data paths
    TRAIN_PATH = "./input/shanghaitech-with-people-density-map/ShanghaiTech/part_B/train_data/"
    TEST_PATH = "./input/shanghaitech-with-people-density-map/ShanghaiTech/part_B/test_data/"
    
    # Training parameters
    BATCH_SIZE = 8
    LEARNING_RATE = 0.00001
    EPOCHS = 100
    
    # Model parameters
    GT_DOWNSAMPLE = 4
    
    # Device configuration
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'