import os
import torch

ROOT_DIR = os. getcwd()
DATA_DIR = os.path.join(ROOT_DIR+'/data/raw')
TRAIN_DIR = os.path.join(ROOT_DIR+'/data/raw/train')
TEST_DIR = os.path.join(ROOT_DIR+'/data/raw/test')
OUTPUT_DIR = ROOT_DIR

MODEL_NAME = 'efficientnetb0'
MODEL_VERSION = 'v2'
MODEL_PATH = f'{OUTPUT_DIR}/{MODEL_NAME}{MODEL_VERSION}.pth'
MODEL_INF_PATH = os.path.join(ROOT_DIR + '/EfficientNetB0.pth')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# test model path
TEST_MODEL_PATH = f'{OUTPUT_DIR}/resnext50_32x4dv2.pth'

# training params
BATCH_SIZE = 36 
EPOCHS = 1
LR = 1e-5
WEIGHT_DECAY = 5e-4

# ensemble model paths
ENS_MODEL_1 = 'resnext50_32x4d'
ENS_MODEL_2 = 'efficientnetb0'
ENS_MODEL_1_PATH = f'{OUTPUT_DIR}/ResNext50.pth'
ENS_MODEL_2_PATH = f'{OUTPUT_DIR}/EfficientNetB0.pth'

#inference
IMG_PATH = os.path.join(ROOT_DIR+'/bird.jpg')

print(IMG_PATH)
