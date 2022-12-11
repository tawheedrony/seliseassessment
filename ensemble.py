import torch
import pandas as pd
import numpy as np
import test
import config
import train
import utils


test_df, class_indices = utils.create_dir_map(root_dir = config.TEST_DIR)
testloader = utils.preprocess_dataloader(class_indices=class_indices, test_df=test_df)

# RESNEXT50
model_1 = config.ENS_MODEL_1
ensnet_1 = train.get_model(model_name = model_1).to(config.DEVICE)
checkpoint_1 = torch.load(config.ENS_MODEL_1_PATH, map_location=config.DEVICE)
ensnet_1.load_state_dict(checkpoint_1)
ytrue_1, ypred_1 = test.test_fn(model=ensnet_1,testloader=testloader)

#EFFNETB0
model_2 = config.ENS_MODEL_2
ensnet_2 = train.get_model(model_name = model_2).to(config.DEVICE)
checkpoint_2 = torch.load(config.ENS_MODEL_2_PATH, map_location=config.DEVICE)
ensnet_2.load_state_dict(checkpoint_2)
ytrue_2, ypred_2 = test.test_fn(model=ensnet_2,testloader=testloader)

# WEIGHTED ENSEMBLE
pred_ens = 0.5 * ypred_1 + 0.5 * ypred_2
print(f"Test accuracy is {round(np.mean(ytrue_1 == pred_ens)*100,4)}%")


