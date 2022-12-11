import utils
import train
import config

import torch

import torch.nn as nn


def run():    
    #dataloader
    main_df, class_indices = utils.create_dir_map(root_dir = config.TRAIN_DIR)
    trainloader, validloader = utils.preprocess_dataloader(main_df=main_df, class_indices=class_indices)
    
    #model 
    model_name = config.MODEL_NAME
    model = train.get_model(model_name = model_name).to(config.DEVICE)

    #loss function
    criterion = torch.nn.CrossEntropyLoss()

    #optimizer
    if model_name == 'resnext50_32x4d':
        params_1x = [param for name, param in model.named_parameters() if 'fc' not in str(name)]
        optimizer = torch.optim.Adam([{'params': params_1x}, 
                                    {'params': model.fc.parameters(), 'lr': config.LR*10}], 
                                    lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    elif model_name == 'efficientnetb0':
        params_1x = [param for name, param in model.named_parameters() if 'classifier' not in str(name)]
        optimizer = torch.optim.Adam([{'params': params_1x}, 
                                    {'params': model.classifier.parameters(), 'lr': config.LR*10}], 
                                    lr=config.LR, weight_decay=config.WEIGHT_DECAY)   
    else:
        print("Optimizer Not Set")
        
    # start training
    model, training_metrics = train.train_fn(model, trainloader, validloader, criterion, 
                                optimizer, scheduler=None, epochs=config.EPOCHS, device=config.DEVICE)
    
    print("DONE")
    
if __name__ == "__main__":
    run()