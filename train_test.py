#%%
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.data.utils import load_graphs

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

# self construct functions
from node_evaluation import collate, evalEdge 
from SAGEE import SAGEE


pd.options.mode.chained_assignment = None  # default='warn'

device = torch.device('cpu') # CPU is enough for processing small graphs
print('Using device:', device)





#%% parameters 
name = "SAGEE" # file name for storing figures 
bg = load_graphs("./graph/roomgraph.bin")[0]
epochs = 200

batch_size = 1

n_classes = 9 # nine room classes here

weight_decay=5e-4
num_channels = 50
lr = 0.005

 

################### Training prepare ##################
#### data split
trainvalid, test_dataset =  train_test_split(bg, test_size=0.2, random_state=42)
train_dataset, valid_dataset = train_test_split(trainvalid, test_size=0.1, random_state=42)

#### data batch for parallel computation
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate)

print("train dataset %i, val dataset %i, test dataset %i"%(len(train_dataset), \
    len(valid_dataset), len(test_dataset)))

#### model loading 
ndim_in = train_dataset[0].ndata['feat'].shape[1]
edim_in = train_dataset[0].edata['relation'].shape[1]

model = SAGEE(ndim_in, n_classes, edim_in,  F.relu, 0.2)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
model = model.to(device)
print(model)


################### Training process ##################

train_acc_all, train_loss_all  = [], []
val_acc_all, val_loss_all = [], []


train_startime = time.time()

for epoch in range(epochs):

    #### train one epoch 
    model.train()

    train_acc_list = []
    train_loss_list = []

    for batch, subgraph in enumerate(train_dataloader):

        subgraph = subgraph.to(device) 
        nfeat = subgraph.ndata['feat'].float()
        efeat = subgraph.edata['relation'].float()

        logits = model(subgraph, nfeat, efeat) # get the prediction from models 

        # calculate the accuracy 
        gt = torch.argmax(subgraph.ndata['label'], dim=1) # ground true labels
        pre  = torch.argmax(logits, dim=1)  # prediction labels 
        correct = torch.sum(pre == gt) # calculate the right labels 

        acc = correct.item()*1.0/len(gt) # calculate the accuracy 
        train_acc_list.append(acc) 

        # compute the loss
        loss = F.cross_entropy(logits, gt) # using cross entropy 
        train_loss_list.append(loss.item()) 

        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss_epoch =  np.array(train_loss_list).mean()
    train_acc_epoch =  np.array(train_acc_list).mean()

    train_loss_all.append(train_loss_epoch)
    train_acc_all.append(train_acc_epoch)
    
    # loss_data = np.array(loss_list).mean()
    print("Epoch {:05d} | Accuracy: {:.4f} | Loss: {:.4f}".format(\
        epoch+1, train_acc_epoch, train_loss_epoch))

    #### evaluate one epoch 
    if epoch % 5 == 0:
        val_acc_list = []
        val_loss_list = []
        class_correct = [0. for i in range(n_classes)]
        class_total = [0. for i in range(n_classes)]
        class_acc = [0. for i in range(n_classes)]


        for batch, subgraph in enumerate(valid_dataloader):
            subgraph = subgraph.to(device)

            # calculate the accuracy and loss
            nfeat = subgraph.ndata['feat'].float()
            efeat = subgraph.edata['relation'].float()

            acc, loss, one_class_correct, one_class_total, _, _ = evalEdge(model, nfeat, efeat, subgraph, subgraph.ndata['label'], n_classes)

            # obtain acc and loss
            val_acc_list.append(acc)
            val_loss_list.append(loss.item())
            # obtian correct_num, total_num
            class_correct = np.sum([class_correct, one_class_correct], axis=0)
            class_total = np.sum([class_total, one_class_total], axis=0)

        # calculate the loss and acc for all graphs in one epoch
        val_loss_epoch =  np.array(val_loss_list).mean()
        val_acc_epoch =  np.array(val_acc_list).mean()

        # calcuate the acc of each class for all graphs in one epoch
        for i in range(n_classes):
            if class_total[i] != 0 :
                class_acc[i] = class_correct[i] / class_total[i]
            else:
                class_acc[i] = -1
        
        # append for drawing the curs
        val_acc_all.append(val_acc_epoch)
        val_loss_all.append(val_loss_epoch)
            
        print("Val | Accuracy: {:.4f} | Loss: {:.4f}\n".format(val_acc_epoch, val_loss_epoch))
        print("Accuracy for each class:\nKitchen:{:.4f} \t| Living:{:.4f} \t| Dining:{:.4f} \nBed:{:.4f} \t| Masterbed:{:.4f} \t| Toilet:{:.4f} \nCloset:{:.4f} \t| Balcony:{:.4f} \t| laundry:{:.4f}\n".format(\
            class_acc[0], class_acc[1], class_acc[2], class_acc[3], \
                class_acc[4], class_acc[5], class_acc[6], class_acc[7], class_acc[8]))

        ############ save the best acc epoch ############ 
        if val_acc_epoch >= val_acc_all:
            torch.save(model, "best.pt")
    
train_endtime = time.time()

print("Finish training! Using {:.4f} s:".format(train_endtime - train_startime))

 



################################# Test based on the best weight ###############################
print("\n\n\nStart to test...")

model = torch.load("best.pt")  # read the best weight
model.eval()    


test_acc_list = [] # list for storing the acc from each graph
 
test_startime = time.time()

for batch, subgraph in enumerate(test_dataloader):
    subgraph = subgraph.to(device)
    # subgraph = dgl.add_self_loop(subgraph)

    nfeat = subgraph.ndata['feat'].float()
    efeat = subgraph.edata['relation'].float()

    acc, _, _, _, _, _ = evalEdge(model, nfeat, efeat, \
        subgraph, subgraph.ndata['label'], n_classes)

    test_acc_list.append(acc)

test_endtime = time.time()
  

test_acc = np.array(test_acc_list).mean()   # get the acc for test 

cm = confusion_matrix(gt, pre)  # confusion matrix, default function from scikit-learn

f1 = f1_score(gt, pre, average='macro') # f1, default function from scikit-learn

print("Test Accuracy: {:.4f}".format(test_acc))
 
print(f"Confusion matrix\n{cm}")

print("F1 score: {:4f}".format(f1))

print("Test time: {:.4f}".format(test_endtime-test_startime))