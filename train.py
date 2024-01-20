import torch.nn as nn
import pandas as pd
from utils import *

def train(train_loader, epoch, \
          model, criterion, optimizer, device
          ):
    model.train()
    train_losses=AverageMeter()
    for i, (input, target, _) in enumerate(train_loader):
        input = input.to(device)
        target = target.to(device)
        output = nn.Sigmoid()(model(input))
        loss = criterion(output,target).float()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.update(loss.detach().cpu().numpy(),input.shape[0])
    Train_Loss=np.round(train_losses.avg,6)
    return Train_Loss
    
def validate(validation_loader, 
          model, criterion, device,
        model_path=False,
             return_image_paths=False,
          ):
    if model_path!=False:
        model.load_state_dict(torch.load(model_path))
    model.eval()
    for i, (input, target, image_path) in enumerate(validation_loader):
        input =input.to(device)
        target = target.to(device)
        with torch.no_grad():
            output = nn.Sigmoid()(model(input))
        if i==0:
            targets=target
            outputs=output
            if return_image_paths==True:
                image_paths = image_path
        else:
            targets=torch.cat((targets,target))
            outputs=torch.cat((outputs,output),axis=0)
            if return_image_paths==True:
                image_paths += image_path
    if return_image_paths==True:
        return outputs, targets, image_paths
    return outputs, targets

class LossSaver(object):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
    def reset(self):
        self.train_losses = []
        self.val_losses = []
    def update(self, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
    def return_list(self):
        return self.train_losses, self.val_losses
    def save_as_csv(self, csv_file):
        df = pd.DataFrame({'Train Losses': self.train_losses, 'Validation Losses': self.val_losses})
        df.index = [f"{i+1} Epoch" for i in df.index]
        df.to_csv(csv_file, index=True)
