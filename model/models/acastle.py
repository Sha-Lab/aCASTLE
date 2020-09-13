import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from ..utils import one_hot

from model.models import GeneralizedFewShotModel

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        # combine local and global key and value
        k = torch.cat([q, k], 1)
        v = torch.cat([q, v], 1)
        len_k = len_k + len_q
        len_v = len_v + len_q        
        
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = output + residual
        return output, attn, log_attn

class ACastle(GeneralizedFewShotModel):
    def __init__(self, args):
        super().__init__(args)
        hdim = self.hdim
        # for acastle module
        self.slf_attn = MultiHeadAttention(args.head, hdim, hdim // args.head, hdim // args.head, dropout=args.dp_rate)   
        # for shared memory
        self.shared_key = torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.zeros(1, args.num_class * 2, hdim)))
        # for the global classifier
        self.cls = nn.Linear(hdim, args.num_class, bias=False)
            
    def _forward(self, support, query, index_label):
        support_idx, support_labels = index_label
        num_task = support_idx.shape[0]
        num_dim = support.shape[-1]
        # organize support data
        support = support[support_idx.view(-1)].view(*(support_idx.shape + (-1,)))
        proto = support.mean(dim=1) # Ntask x NK x d
        num_proto = proto.shape[1]     
        logit = []
        for tt in range(num_task):
            # combine proto with the global classifier
            global_mask = torch.eye(self.args.num_class)
            if torch.cuda.is_available():
                global_mask = global_mask.cuda()  
            whole_support_index = support_labels[tt,:]
            global_mask[:, whole_support_index] = 0
            # construct local mask
            local_mask = one_hot(whole_support_index, self.args.num_class)
            current_classifier = torch.mm(self.cls.weight.t(), global_mask) + torch.mm(proto[tt,:].t(), local_mask)            
            # regroup the classifier based on tasks
            current_classifier = current_classifier.t()
            # training with generalized classification (multiple tasks)       
            cls, _, _ = self.slf_attn(current_classifier.unsqueeze(0), self.shared_key, self.shared_key)
            cls = cls.squeeze(0).t()
            cls = F.normalize(cls, dim=0)
            logit.append(torch.mm(query, cls))
        logit = torch.cat(logit, 1)
        logit = logit.view(-1, self.args.num_class)        
        
        ## matrix implementation
        ## combine proto with the global classifier
        #global_mask = torch.eye(self.args.num_class).repeat(1, num_task)
        #global_mask_basis = torch.arange(num_task).long().view(-1,1).repeat(1, num_proto) * self.args.num_class
        #if torch.cuda.is_available():
            #global_mask = global_mask.cuda()  
            #global_mask_basis = global_mask_basis.cuda()
        #whole_support_index = (support_labels + global_mask_basis).view(-1)
        #global_mask[:, whole_support_index] = 0
        ## construct local mask
        #local_mask = one_hot(whole_support_index, self.args.num_class * num_task)
        #proto = proto.view(-1, num_dim)
        #current_classifier = torch.mm(self.cls.weight.t(), global_mask) + torch.mm(proto.t(), local_mask)            
        ## regroup the classifier based on tasks
        #current_classifier = current_classifier.t()
        #current_classifier = current_classifier.view(self.args.num_task, self.args.num_class, num_dim)
        ## training with generalized classification (multiple tasks)       
        #cls, _, _ = self.slf_attn(current_classifier, self.shared_key.repeat(num_task, 1, 1), self.shared_key.repeat(num_task, 1, 1)) # num_task * num_way * dim
        #cls = cls.view(-1, num_dim)
        #cls = cls.t()
        #cls = F.normalize(cls, dim=0)
        #logit = torch.mm(query, cls)
        #logit = logit.view(-1, self.args.num_class)      
        return logit   

    def _forward_fsl(self, support, query, aux=None):
        channel = support.shape[-1]
        proto = support.reshape(self.args.eval_shot, -1, channel).mean(dim=0) # N x d
        cls, _, _ = self.slf_attn(proto.unsqueeze(0), self.shared_key, self.shared_key)
        cls = cls.squeeze(0)
        cls = F.normalize(cls, dim=1)      
        # num_proto = proto.shape[0]
        # cls = cls[-1 - num_proto:-1, :]
        logit = torch.mm(query, cls.t())  
        return logit   

    def _forward_gfsl(self, support_embs, query_embs, aux=None):
        num_dim = support_embs.shape[-1]
        proto = support_embs.reshape(self.args.eval_shot, -1, num_dim).mean(dim=0) # N x d
        current_classifier = torch.cat([self.cls.weight, proto], 0)       
        cls, _, _ = self.slf_attn(current_classifier.unsqueeze(0), self.shared_key, self.shared_key)
        cls = cls.squeeze(0)
        cls = F.normalize(cls, dim=1)      
        cls_seen = cls[:self.args.num_class, :]
        cls_unseen = cls[self.args.num_class:, :]
            
        logits_s = torch.mm(query_embs, cls_seen.t()) / self.args.temperature
        logits_u = torch.mm(query_embs, cls_unseen.t()) / self.args.temperature
        return logits_s, logits_u