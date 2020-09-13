import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from ..utils import one_hot
from model.utils import euclidean_metric

from model.models import GeneralizedFewShotModel

class Classifier(GeneralizedFewShotModel):
    def __init__(self, args):
        super().__init__(args)
        self.fc = nn.Linear(self.hdim, args.num_class)
        
    def _forward_fsl(self, support_embs, query_embs, aux=None):
        num_dim = support_embs.shape[-1]
        unseenproto = support_embs.reshape(self.args.eval_shot, -1, num_dim).mean(dim=0) # N x d        
        logits_u = euclidean_metric(query_embs, unseenproto)
        return logits_u  

    def _forward_gfsl(self, support_embs, query_embs, aux=None):
        num_dim = support_embs.shape[-1]
        unseenproto = support_embs.reshape(self.args.eval_shot, -1, num_dim).mean(dim=0) # N x d
        
        logits_s = self.fc(query_embs)
        logits_u = euclidean_metric(query_embs, unseenproto)
        return logits_s, logits_u
    
    def forward_many(self, data):
        return self.fc(self.encoder(data))