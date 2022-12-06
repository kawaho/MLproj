import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

def KL(alp_til, rank):
    ones = torch.Tensor(np.ones((1,10))).to(rank)

    S_alpha = torch.sum(alp_til, 1, keepdim=True)
    S_ones = torch.sum(ones, 1, keepdim=True)

    lnD = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alp_til), 1, keepdim=True)
    lnD_uni = torch.sum(torch.lgamma(ones), 1, keepdim=True) - torch.lgamma(S_ones)
    
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alp_til)
    
    kl = torch.sum((alp_til - ones)*(dg1 - dg0), 1, keepdim=True) + lnD + lnD_uni
    return kl

def mseKL_loss(alpha, truth, epoch, rank, mean=True): 
    S = torch.sum(alpha, 1, keepdim=True) 
    E = alpha - 1
    pred = alpha / S
    err = torch.sum((truth-pred)**2, 1, keepdim=True) 
    var = torch.sum(alpha*(S-alpha)/(S*S*(S+1)), 1, keepdim=True) 
    
    annealing_coef = min(1.0, epoch/10)
    
    alp_til = E*(1-truth) + 1 
    penalty =  annealing_coef * KL(alp_til, rank)
    if mean:
      return torch.mean(err + var + penalty)
    return torch.sum(err + var + penalty)
    

class model_softm(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)
        #return F.log_softmax(x, dim=1)

class model_diri(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return torch.exp(torch.clamp(self.fc2(x), min=-10, max=10))
        #return F.relu(self.fc2(x))
