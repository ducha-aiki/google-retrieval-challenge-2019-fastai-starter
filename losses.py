from fastprogress import master_bar, progress_bar
#import matplotlib.pyplot as plt
from fastai.vision import *
from fastai.metrics import accuracy
from fastai.basic_data import *
#from skimage.util import montage
import pandas as pd
from torch import optim
import re
import torch
from fastai import *
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn



@dataclass
class RingLoss(Callback):
    learn:Learner
    alpha:float=0.01
    def on_loss_begin(self, last_output:Tuple[tensor,tensor], **kwargs):
        "Save the extra outputs for later and only returns the true output."
        self.feature_out = last_output[1]
        return {'last_output':last_output[0]}

    def on_backward_begin(self,
                          last_loss:Rank0Tensor,
                          **kwargs):
        
        x = self.feature_out
        R = self.learn.model.head.ring
        loss = None
        cc=0
        x_norm = x.pow(2).sum(dim=1).pow(0.5)
        diff = torch.mean(torch.abs(x_norm - R.expand_as(x_norm))**2)
        if loss is None:
            loss = diff.mean()
        else:
            loss = loss + diff.mean()
        if self.alpha != 0.:  last_loss += (self.alpha * loss).sum()
        return {'last_loss':last_loss}
