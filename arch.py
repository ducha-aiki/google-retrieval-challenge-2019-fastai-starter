#https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/cirtorch/layers/pooling.py
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from fastai import *
from fastai.vision import Flatten

def gem(x, p=3, eps=1e-5):
    return torch.abs(F.avg_pool2d(x.clamp(min=eps, max=1e4).pow(p), (x.size(-2), x.size(-1))).pow(1./p))

class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(1).expand_as(x)
        return x

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=torch.clamp(self.p, min=0.1), eps=self.eps)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class RingHead(nn.Module):
    def __init__(self, num_classes, feat_dim, in_feat = 1024, r_init =1.5):
        super(RingHead,self).__init__()
        self.eps = 1e-10
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.feature_extractor = nn.Sequential(
                        nn.ReLU(),
                        GeM(3.74), Flatten(),
                        nn.BatchNorm1d(in_feat, eps=1e-05, momentum=0.1,
                                                       affine=True, track_running_stats=True),
                        nn.Dropout(p=0.3),
                        nn.Linear(in_features=in_feat, out_features=feat_dim, bias=True),
                        nn.CELU(inplace=True),
                        nn.BatchNorm1d(feat_dim,eps=1e-05, momentum=0.1,
                                                       affine=True, track_running_stats=True))
        
        self.ring =  nn.Parameter(torch.ones(1).cuda()*r_init)
        self.clf = nn.Sequential(nn.Dropout(p=0.5),
                        nn.Linear(in_features=feat_dim, out_features=num_classes, bias=False))
    def forward(self, x):
        feats = self.feature_extractor(x)
        preds = self.clf(feats)
        return preds,feats

class RingGeMNet(nn.Module):
    def __init__(self, new_model, n_classes, in_feats=1024, out_feats=1024):
        super().__init__()
        self.cnn =  new_model.features
        self.head = RingHead(n_classes, out_feats, in_feats)
    def forward(self, x):
        x = self.cnn(x)
        preds,feats = self.head(x)
        return preds,feats

class GeMNet(nn.Module):
    def __init__(self, new_model):
        super().__init__()
        self.cnn =  new_model.features
        self.head = nn.Sequential(nn.ReLU(),
                                  GeM(5.0),
                                  Flatten(),
                                  L2Norm())
    def forward(self, x):
        x = self.cnn(x)
        out = self.head(x)
        return out
