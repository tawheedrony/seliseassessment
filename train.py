import torch
import torchvision
import time
import config

def get_model(model_name):
    if model_name == 'resnext50_32x4d':
        model = torchvision.models.resnext50_32x4d(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 4)
        torch.nn.init.xavier_uniform_(model.fc.weight)
    elif model_name == 'efficientnetb0':
        model = torchvision.models.efficientnet_b0(pretrained=True)
        model.classifier = torch.nn.Linear(1280, 4)
        torch.nn.init.xavier_uniform_(model.classifier.weight)
    else:
        print("Accurate/No model stated")
    return model


def train_fn(net, train_dataloader, valid_dataloader, criterion, optimizer, scheduler=None, epochs=10, device='cpu'):
    
    start = time.time()
    print(f'Training for {epochs} epochs on {device}')
    
    save_path = config.MODEL_PATH
    best_val_acc = 0.0
    training_metrics = []
    for epoch in range(1,epochs+1):
        print(f"Epoch {epoch}/{epochs}")
        
        net.train()  
        train_loss = torch.tensor(0., device=device)  
        train_accuracy = torch.tensor(0., device=device)
        for X, y in train_dataloader:
            X = X.to(device)
            y = y.to(device)
            preds = net(X)
            loss = criterion(preds, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                train_loss += loss * train_dataloader.batch_size
                train_accuracy += (torch.argmax(preds, dim=1) == y).sum()
        
        if valid_dataloader is not None:
            net.eval()  
            valid_loss = torch.tensor(0., device=device)
            valid_accuracy = torch.tensor(0., device=device)
            with torch.no_grad():
                for X, y in valid_dataloader:
                    X = X.to(device)
                    y = y.to(device)
                    preds = net(X)
                    loss = criterion(preds, y)

                    valid_loss += loss * valid_dataloader.batch_size
                    valid_accuracy += (torch.argmax(preds, dim=1) == y).sum()
        
        if scheduler is not None: 
            scheduler.step()
        
        train_loss = train_loss/len(train_dataloader.dataset)
        train_accuracy = 100*train_accuracy/len(train_dataloader.dataset)
       
        if valid_dataloader is not None:
            valid_loss = valid_loss/len(valid_dataloader.dataset)
            valid_accuracy = 100*valid_accuracy/len(valid_dataloader.dataset)

        
        print(f'Train loss: {train_loss:.3f} Valid loss: {valid_loss:.3f} Train accuracy: {train_accuracy:.3f} Valid accuracy: {valid_accuracy:.3f}')
        metric_dict = {
            'epoch' : epoch,
            'train_loss' : train_loss.cpu().numpy(),
            'train_accuracy' : train_accuracy.cpu().numpy(),
            'valid_loss' : valid_loss.cpu().numpy(),
            'valid_accuracy' : valid_accuracy.cpu().numpy()
        }
        training_metrics.append(metric_dict)
        
        if valid_accuracy > best_val_acc:
            best_val_acc = valid_accuracy
            torch.save(net.state_dict(), save_path)
    
    end = time.time()
    print(f'Total training time: {end-start:.1f} seconds')
    return net, training_metrics




