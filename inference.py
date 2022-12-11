import torch
import config
import train
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def inference():
    model_name = config.MODEL_NAME
    model = train.get_model(model_name = model_name).to(config.DEVICE)
    checkpoint = torch.load(config.MODEL_INF_PATH, map_location=config.DEVICE)
    model.load_state_dict(checkpoint)
    
    class_dict = {'berry': 0, 'bird': 1, 'dog': 2, 'flower': 3}
    img = Image.open(config.IMG_PATH)
    inf_transform = transforms.Compose([transforms.Resize((256,256)), 
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                              [0.229, 0.224, 0.225])])
    img_ = inf_transform(img).float().unsqueeze_(0).to(config.DEVICE)
    print(img_.shape)
    with torch.no_grad():
        model.eval()  
        output =model(img_)
        index = output.data.cpu().numpy().argmax()
        classs = {i for i in class_dict if class_dict[i]==index}
        print(classs)
        return classs
    
if __name__ == "__main__":
    inference()