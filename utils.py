import pandas as pd 
import numpy as np
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from torchvision import datasets, transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os 
import random
import warnings
warnings.filterwarnings("ignore")

import config
    
class SeliseDataset(Dataset):
    def __init__(self, image_path, image_labels, transform=None) -> None:
        super().__init__()
        self.image_path = image_path
        self.image_class = image_labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, item):
        img = Image.open(self.image_path[item]).convert('RGB')
        label = self.image_class[item]
        
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels   




def create_dir_map(root_dir):
    filepaths = []
    labels = []
    
    
    classes = [cla for cla in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, cla))]
    classes.sort()
    class_indices = {k: v for v, k in enumerate(classes)}
    
    every_class_num = []
    for klass in classes:
        classpath = os.path.join(root_dir, klass)
        images = [os.path.join(root_dir, klass, i) for i in os.listdir(classpath)]
        every_class_num.append(len(images))
        flist = sorted(os.listdir(classpath))
        for (index,f) in enumerate(flist):
            fpath = os.path.join(classpath, f)
            img = cv2.imread(fpath)
            filepaths.append(fpath)
            labels.append(klass)

    Fseries = pd.Series(filepaths, name='filepaths')
    Lseries = pd.Series(labels, name='labels')
    df = pd.concat([Fseries, Lseries], axis=1)
    return df, class_indices


def preprocess_dataloader(class_indices, main_df=None, test_df=None):
 
    data_transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(256), 
        transforms.ToTensor(), 
        transforms.RandomHorizontalFlip(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])}
    

    if main_df is not None:
            
        train_df, val_df = train_test_split(main_df, train_size=.8, shuffle=True,
                                        random_state=123, stratify=main_df['labels'])

        train_image_path = train_df['filepaths'].tolist()
        val_image_path = val_df['filepaths'].tolist()
        
        train_image_label = [class_indices[i] for i in train_df['labels'].tolist()]
        val_image_label = [class_indices[i] for i in val_df['labels'].tolist()]
        train_dataset = SeliseDataset(train_image_path, train_image_label, 
                                transform=data_transform['train'])   
        valid_dataset = SeliseDataset(val_image_path, val_image_label, 
                                transform=data_transform['valid'])
        
        
        
        trainloader = DataLoader(train_dataset, shuffle=True, 
                                batch_size=config.BATCH_SIZE, num_workers=0, 
                                collate_fn=train_dataset.collate_fn)
        validloader = DataLoader(valid_dataset, shuffle=False, 
                                batch_size=config.BATCH_SIZE, num_workers=0, 
                                collate_fn=valid_dataset.collate_fn)   
        
        return trainloader, validloader     
    
    if test_df is not None:
        test_image_path = test_df['filepaths'].tolist()
        test_image_label = [class_indices[i] for i in test_df['labels'].tolist()]
        test_dataset = SeliseDataset(test_image_path, test_image_label, 
                            transform=data_transform['valid'])   
        testloader = DataLoader(test_dataset, shuffle=False, 
                            batch_size=config.BATCH_SIZE, num_workers=0, 
                            collate_fn=test_dataset.collate_fn)
        return  testloader
    
    




