import numpy as np
import torch
import dgl
import torch.nn.functional as F




# collect single small graphs as a batch
def collate(graphs):
    graph = dgl.batch(graphs)
    return graph




def evalEdge(model, nfeat, efeat, subgraph, labels, n_classes):
    "This function can be fed with node feature and edge feature, output the prediction of the model"

    with torch.no_grad():
        model.eval()
        # subgraph = subgraph.to(device)

        # output the prediction results
        logits = model(subgraph, nfeat, efeat) 

        # calculate the accuracy
        gt = torch.argmax(labels,dim=1) #labels = subgraph.ndata['label']
        pre  = torch.argmax(logits, dim=1)
        correct = torch.sum(pre == gt)
        acc = correct.item()*1.0/len(gt)        

        # compute the loss
        loss = F.cross_entropy(logits, gt)
        
        # statistic the correct predictions numbers of each class
        one_class_correct, one_class_total = accEachClass(gt, pre, n_classes)
        
        return acc, loss, one_class_correct, one_class_total, pre, gt
        
def accEachClass(gt, pre, n_classes):
    one_class_correct = list(0. for i in range(n_classes))
    one_class_total = list(0. for i in range(n_classes))
    # class_acc = list(0. for i in range(n_classes))

    for i in range(len(gt)):
        # for each correct prediction, +1
        if gt[i] == pre[i]:
            one_class_correct[gt[i]] += 1

        one_class_total[gt[i]] += 1 
    
    return one_class_correct, one_class_total
