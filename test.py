import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import config
import train
import utils




def test_predictor():
    
    test_df, class_indices = utils.create_dir_map(root_dir = config.TEST_DIR)
    testloader = utils.preprocess_dataloader(class_indices=class_indices, test_df=test_df)
    
    model_name = config.MODEL_NAME
    model = train.get_model(model_name = model_name).to(config.DEVICE)
    
    errors = 0
    y_pred, y_true = [], []
    checkpoint = torch.load(config.TEST_MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(checkpoint)

    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            for i in range(len(preds)):
                y_pred.append(preds[i].cpu())
                y_true.append(labels[i].cpu())
    
    tests = len(y_pred)
    for i in range(tests):
        pred_index = y_pred[i]
        true_index = y_true[i]
        if pred_index != true_index:
            errors += 1
    acc = (1 - errors / tests) * 100
    print(f'there were {errors} errors in {tests} tests for an accuracy of {acc:6.2f}%')
    ypred = np.array(y_pred)
    ytrue = np.array(y_true)



def test_fn(model,testloader):
    y_pred, y_true = [], []
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            for i in range(len(preds)):
                y_pred.append(preds[i].cpu())
                y_true.append(labels[i].cpu())
                
    ypred = np.array(y_pred)
    ytrue = np.array(y_true)
    return ytrue, ypred
    

if __name__ == "__main__":
    test_predictor()

    
    
    
    
#ytrue, ypred = test_predictor(testloader, net)

#Print classification report

#print(f"Test accuracy is {round(np.mean(ytrue == ypred)*100,4)}%")
#print('F1 score is',round(f1_score(ytrue,ypred, average = 'weighted') *100,4), "%")

#print("Confusion Matrix Heatmap")
#cf_matrix = confusion_matrix(ytrue,ypred )
#df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix,axis=1),
#                     index = [i for i in class_indices],
#                     columns = [i for i in class_indices])
#plt.figure(figsize = (8,5))
#sns.heatmap(df_cm, fmt='.2%',annot=True, cmap='Blues')

