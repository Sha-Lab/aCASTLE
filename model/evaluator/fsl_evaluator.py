import time
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F

from model.evaluator.helpers import (
    get_dataloader, prepare_model,
)
from model.utils import (
    pprint, ensure_path,
    Averager, Timer, count_acc, one_hot, compute_confidence_interval
)
from tqdm import tqdm

class FSLEvaluator():
    def __init__(self, args):
        super().__init__()
        args.temperature = 1.0
        self.args = args
        # train statistics
        self.trlog = {}

        self.trainset, self.valset, self.trainvalset, self.testset, self.traintestset, \
            self.train_fsl_loader, self.train_gfsl_loader, self.train_proto_loader, self.val_fsl_loader, self.val_gfsl_loader, self.test_fsl_loader, self.test_gfsl_loader = get_dataloader(args)
        self.model = prepare_model(args)
        self.model.eval()
       
    def evaluate_fsl(self):
        args = self.args        
        record = np.zeros((args.num_eval_episodes, 2)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
            
        for i, batch in tqdm(enumerate(self.test_fsl_loader, 1)):
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            p2 = args.eval_shot * args.eval_way
            
            p = args.eval_shot * args.eval_way
            data_shot, data_query = data[:p], data[p:]
            with torch.no_grad():
                logits = self.model.forward_fsl(data_shot, data_query)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            record[i-1, 0] = loss.item()
            record[i-1, 1] = acc     
            
            del loss, logits
            torch.cuda.empty_cache()
        assert(i == record.shape[0])   
        
        print('-'.join([args.model_class, args.model_path]))
        self.trlog['acc_mean'], self.trlog['acc_interval'] = compute_confidence_interval(record[:,1])             
        print('FSL {}-way Acc {:.5f} + {:.5f}'.format(args.eval_way, self.trlog['acc_mean'], self.trlog['acc_interval']))
            
    def final_record(self):
        # save the best performance in a txt file
        save_path = osp.join(*self.args.model_path.split('/')[:-1])
        with open(osp.join(save_path, '{}-{}-way: {}+{}.txt'.format('FSL', self.args.eval_way, self.trlog['acc_mean'], self.trlog['acc_interval'])), 'w') as f:
            f.write('FSL {}-way Accuracy {:.5f} + {:.5f}\n'.format(self.args.eval_way, self.trlog['acc_mean'], self.trlog['acc_interval']))
            