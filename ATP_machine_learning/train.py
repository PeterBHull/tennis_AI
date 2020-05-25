import numpy as np
import pandas as pd
import pyarrow
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import sklearn
from sklearn.model_selection import train_test_split as split
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import torch.optim as optim
import os
import logging
import copy
class binaryClassification(nn.Module):
    def __init__(self, layer_width, layer_depth,input_size, dropout = 0, activation = 'ReLU', final_activation = 'sigmoid', ):
        super(binaryClassification, self).__init__()
        self.activation = activation
        self.final_act = final_activation
        self.drop_layer = nn.Dropout(p=dropout)

        self.linears = nn.ModuleList([(nn.Linear(input_size, layer_width))])
        self.linears.extend([nn.Linear(layer_width, layer_width) for i in range(1, layer_depth-1)])
        self.linears.append(nn.Linear(layer_width, 1))

    def forward(self, x):
        for i in range(len(self.linears)-1):
            if self.activation == 'ReLU':
                x = self.drop_layer(F.relu(self.linears[i](x)))
            else:
                sys.exit('Bad activation function string')
            x = self.drop_layer(F.sigmoid(self.linears[i+1](x)))
        return x

class Experiment:
    def __init__(self,experiment_name, project_name, layer_width, layer_depth, batch_size, n_splits = 8, device = 'gpu', lr = 0.001):
        self.project_name = project_name
        self.layer_width = layer_width
        self.layer_depth = layer_depth
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.n_splits = n_splits
        self.kf = KFold(n_splits = self.n_splits, random_state = 2)
        self.lr = lr
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.init_experiment_name = experiment_name
        torch.cuda.set_device(self.device)

    def load_data(self, directory_path):
        self.train_x = pd.read_parquet(os.path.join(directory_path,'trainx.parquet'))
        self.train_y = pd.read_parquet(os.path.join(directory_path,'trainy.parquet'))
        self.column_names = self.train_x.columns
        idx = np.random.permutation(self.train_x.index)
        self.train_x.reindex(idx)
        self.train_y.reindex(idx)

        self.train_x = self.train_x.values
        self.train_y = self.train_y.values
        self.input_size = self.train_x.shape[1]
        


    def train(self, loss_type = "BCE"):
        '''
        CROSS VALIDATION TRAINING
        '''
        self.criterion = nn.BCELoss()

        counter = 1
        for train_index, test_index in self.kf.split(self.train_x, self.train_y):
            flag = True
            x_train_fold = torch.tensor(self.train_x[train_index]).float().to(self.device)
            y_train_fold = torch.tensor(self.train_y[train_index]).float().to(self.device)
            x_val_fold = torch.tensor(self.train_x[test_index]).float().to(self.device)
            y_val_fold = torch.tensor(self.train_y[test_index]).float().to(self.device)
            train = torch.utils.data.TensorDataset(x_train_fold,y_train_fold)
            train_loader = torch.utils.data.DataLoader(train, batch_size = self.batch_size, shuffle=True)
            self.experiment_name = self.project_name + self.init_experiment_name + f'Depth:{self.layer_depth}Width:{self.layer_width}n_splits{self.n_splits}lr:{self.lr}kfold:{counter}'

            if counter == 1:
                #if is first fold then need to initialize model and save initial weights
                self.net = binaryClassification(layer_width = self.layer_width,layer_depth = self.layer_depth,input_size = self.input_size)
                self.net.to(self.device)
                self.init_weights = copy.deepcopy(self.net.state_dict())
            else:
                del self.net  
                self.net = binaryClassification(layer_width = self.layer_width,layer_depth = self.layer_depth,input_size = self.input_size)
                self.net.load_state_dict(self.init_weights)
                self.net.to(self.device)

            wandb.init(project = self.project_name, name = self.experiment_name, reinit = True)
            wandb.watch(self.net)
            
            wandb.config.layer_width = self.layer_width
            wandb.config.layer_depth = self.layer_depth
            wandb.config.batch_size = self.batch_size
            wandb.config.n_splits = self.n_splits
            wandb.config.input_size = self.input_size
            wandb.config.batch_size = self.batch_size

            
            #reset optimizer
            self.opt = optim.AdamW(self.net.parameters(), lr=self.lr, betas=(0.9, 0.999))
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max = 7500)
            self.steps = 0
            self.val_loss = 0
            self.train_loss = 0
            self.accuracy = 0
            self.lowest_val_loss = 1000
            self.lowest_val_loss_step = 1
            while flag:
                for i, (x_batch, y_batch) in enumerate(train_loader):
                    
                    self.net.train()
                    #reset gradients
                    self.opt.zero_grad()
                    #Forward
                    y_hat = self.net(x_batch)

                    #training accuracy
                    self.train_accuracy = accuracy_score(np.round(y_hat.cpu().detach().numpy()),np.round(y_batch.cpu().detach().numpy()))

                    #Compute loss
                    loss = self.criterion(y_hat, y_batch)
                    self.train_loss = loss
                    #Compute gradients
                    loss.backward()
                    #update weights
                    if self.device != "cpu":
                        torch.cuda.synchronize()
                    self.opt.step()
                    self.steps += 1
                    logging.warning(f"step {self.steps}")
                    #set net to evaluation mode for validation loss
                    
                    self.net.eval()
                    y_val_hat = self.net(x_val_fold)
                    self.val_loss = self.criterion(y_val_hat,y_val_fold)
                    print(f'lowest_val_loss{self.lowest_val_loss},current_val_loss{self.val_loss}')
                    self.val_accuracy = accuracy_score(np.round(y_val_hat.cpu().detach().numpy()),np.round(y_val_fold.cpu().detach().numpy()))

                    wandb.log({'train_loss': self.train_loss, 'val_loss':self.val_loss, 'val_accuracy':self.val_accuracy,'train_accuracy':self.train_accuracy,'steps':self.steps, 'lr':self.get_lr()})

                    if self.val_loss < self.lowest_val_loss:
                        self.lowest_val_loss = self.val_loss
                        self.lowest_val_loss_step = self.steps
                        self.lowest_val_loss_accuracy = self.val_accuracy
                        self.save_model(counter)
                        wandb.log({'lowest_val_loss': self.lowest_val_loss, 'lowest_val_loss_step': self.steps, 'lowest_val_loss_accuracy': self.lowest_val_loss_accuracy})
                    else:
                        if self.steps >= 7500:  #when to stop training
                            flag = False
                    self.scheduler.step()
            counter += 1
            wandb.join()

    def save_model(self, fold):
        checkpt = { 'init_weights':self.init_weights,
                    'column_names':self.column_names,
                  'state_dict':self.net.state_dict(),
                  'layer_width':self.layer_width,
                  'layer_depth':self.layer_depth,
                  'lowest_val_loss': self.lowest_val_loss,
                  'lowest_val_loss_accuracy': self.lowest_val_loss_accuracy}
        torch.save(checkpt,f'{self.experiment_name}-knum:{fold}-steps:{self.steps}.pt')
        #remove excess files
        l = [k for k in os.listdir() if k.startswith(self.experiment_name)]
        while len(l) >1:
            files = sorted(l, key=os.path.getmtime)
            os.remove(files[0])
            l = [k for k in os.listdir() if k.startswith(self.experiment_name)]
            print(l)
    def get_lr(self):
        for param_group in self.opt.param_groups:
            return param_group['lr']

if __name__=="__main__":
    e = Experiment(project_name = 'tennis_prediction6-cosineanneal',experiment_name = 'fulldata',layer_width = 100, layer_depth = 4, n_splits = 8, device = 'gpu', batch_size = 1024, lr = 0.0001)
    e.load_data('./cleandata/')
    e.train()
