import torch
import torch.nn as nn

class KLDLoss(nn.Module):
    def __init__(self, ):
        super(KLDLoss, self).__init__()
    
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
