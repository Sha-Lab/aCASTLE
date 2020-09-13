import torch
import torch.nn as nn
import numpy as np

def sample_task_ids(support_label, num_task, num_shot, num_way, num_class):
    basis_matrix = torch.arange(num_shot).long().view(-1, 1).repeat(1, num_way).view(-1) * num_class
    permuted_ids = torch.zeros(num_task, num_shot * num_way).long()
    permuted_labels = []
    for i in range(num_task):
        clsmap = torch.randperm(num_class)[:num_way]
        permuted_labels.append(support_label[clsmap])
        permuted_ids[i, :].copy_(basis_matrix + clsmap.repeat(num_shot))      

    return permuted_ids, permuted_labels

class GeneralizedFewShotModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone_class == 'ConvNet':
            hdim = 64
            from model.networks.convnet import ConvNet
            self.encoder = ConvNet()
        elif args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12 import ResNet
            self.encoder = ResNet(dropblock_size=args.dropblock_size)     
        elif args.backbone_class == 'ResNet':
            hdim = 640
            from model.networks.resnet import ResNet
            self.encoder = ResNet()
        else:
            raise ValueError('')
        self.hdim = hdim
        
    def split_instances(self, support_label):
        args = self.args
        if self.training:
            permuted_ids, permuted_labels = sample_task_ids(support_label, args.num_tasks, args.shot, args.way, args.sample_class)
            index_label=(permuted_ids.view(args.num_tasks, args.shot, args.way), torch.stack(permuted_labels))
        else:
            permuted_ids, permuted_labels = sample_task_ids(support_label, args.num_tasks, args.eval_shot, args.eval_way, args.sample_class)
            index_label=(permuted_ids.view(args.num_tasks, args.eval_shot, args.eval_way), torch.stack(permuted_labels))
        return index_label

    def forward(self, x_shot, x_query, shot_label):
        support_emb = self.encoder(x_shot)
        query_emb = self.encoder(x_query)
        index_label = self.split_instances(shot_label)
        logits = self._forward(support_emb, query_emb, index_label)
        return logits
    
    def forward_fsl(self, x_shot, x_query, aux=None):
        # for few-shot learning (during evaluation)
        support_emb, query_emb = [], [] 
        num_support, num_query = x_shot.shape[0], x_query.shape[0]
        for i in range(0, num_support, 128):
            support_emb.append(self.encoder(x_shot[i:min(i+128, num_support), :]))
        support_emb = torch.cat(support_emb, 0)        
        assert(support_emb.shape[0] == num_support)        
        
        for i in range(0, num_query, 128):
            query_emb.append(self.encoder(x_query[i:min(i+128, num_query), :]))
        query_emb = torch.cat(query_emb, 0)        
        assert(query_emb.shape[0] == num_query)        
        
        logits = self._forward_fsl(support_emb, query_emb, aux)
        return logits    
    
    def forward_generalized(self, x_shot, x_query, aux=None):
        # for generalized few-shot learning (during evaluation)
        shot_emb = self.encoder(x_shot)  
        logits_s, logits_u = [], []
        num_query = x_query.shape[0]
        for i in range(0, num_query, 128):
            current_query = self.encoder(x_query[i:min(i+128, num_query), :])
            logit_s, logit_u = self._forward_gfsl(shot_emb, current_query, aux)
            logits_s.append(logit_s) 
            logits_u.append(logit_u)
        logits_s = torch.cat(logits_s, 0)
        logits_u = torch.cat(logits_u, 0)
        assert(logits_s.shape[0] == num_query)
        assert(logits_u.shape[0] == num_query)
        return logits_s, logits_u   

    def _forward(self, x_shot, x_query, idx):
        raise NotImplementedError('Suppose to be implemented by subclass')
    
    def _forward_fsl(self, x_shot, x_query):
        raise NotImplementedError('Suppose to be implemented by subclass')    
    
    def _forward_gfsl(self, support_embs, query_embs, aux=None):
        raise NotImplementedError('Suppose to be implemented by subclass')    