import time
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F

from model.trainer.base import Trainer
from model.trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer,
)
from model.utils import (
    pprint, ensure_path,
    Averager, Timer, count_acc, one_hot,
    compute_confidence_interval, count_acc_harmonic_low_shot_joint
)
from tensorboardX import SummaryWriter
from tqdm import tqdm

class GFSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.trainset, self.valset, self.trainvalset, self.testset, self.traintestset, \
            self.train_fsl_loader, self.train_gfsl_loader, self.val_fsl_loader, self.val_gfsl_loader, self.test_fsl_loader, self.test_gfsl_loader = get_dataloader(args)
        assert(len(self.train_gfsl_loader) == len(self.train_fsl_loader))
        if self.val_gfsl_loader is not None:
            assert(len(self.val_gfsl_loader) == len(self.val_fsl_loader))
        if self.test_gfsl_loader is not None:
            assert(len(self.test_gfsl_loader) == len(self.test_fsl_loader))
        
        self.model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args, len(self.train_gfsl_loader))
        
        self.max_steps = len(self.train_fsl_loader) * args.max_epoch
    
    def train(self):
        args = self.args      
        # start GFSL training
        for epoch in range(1, args.max_epoch + 1):
            self.train_epoch += 1
            self.model.train()
            
            tl1 = Averager()
            tl2 = Averager()
            ta = Averager()

            start_tm = time.time()
            for _, batch in enumerate(zip(self.train_fsl_loader, self.train_gfsl_loader)):
                self.train_step += 1

                if torch.cuda.is_available():
                    support_data, support_label = batch[0][0].cuda(), batch[0][1].cuda()
                    query_data, query_label = batch[1][0].cuda(), batch[1][1].cuda()
                else:
                    support_data, support_label = batch[0][0], batch[0][1]
                    query_data, query_label = batch[1][0], batch[1][1]
                
                data_tm = time.time()
                self.dt.add(data_tm - start_tm)
                
                logits = self.model(support_data, query_data, support_label)
                loss = F.cross_entropy(logits, query_label.view(-1,1).repeat(1, args.num_tasks).view(-1))
                tl2.add(loss.item())
                
                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)
                acc = count_acc(logits, query_label.view(-1,1).repeat(1, args.num_tasks).view(-1))

                tl1.add(loss.item())
                ta.add(acc)

                self.optimizer.zero_grad()
                loss.backward()
                backward_tm = time.time()
                self.bt.add(backward_tm - forward_tm)

                self.optimizer.step()
                self.lr_scheduler.step()
                optimizer_tm = time.time()
                self.ot.add(optimizer_tm - backward_tm)
                self.try_logging(tl1, tl2, ta)               

                # refresh start_tm
                start_tm = time.time()
                del logits, loss
                torch.cuda.empty_cache()                
                
                self.try_evaluate(epoch)

            print('ETA:{}/{}'.format(
                    self.timer.measure(),
                    self.timer.measure(self.train_epoch / args.max_epoch))
            )

        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')

    def evaluate_fsl(self, data_loader):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.eval()
        record = np.zeros((args.num_eval_episodes, 2)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('{} best epoch {}, best val acc={:.4f} + {:.4f}'.format(
            args.test_mode,
            self.trlog['max_acc_epoch'],
            self.trlog['max_acc'],
            self.trlog['max_acc_interval']))
        with torch.no_grad():
            for i, batch in tqdm(enumerate(data_loader, 1)):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                p = args.eval_shot * args.eval_way
                data_shot, data_query = data[:p], data[p:] 
                logits = self.model.forward_fsl(data_shot, data_query)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        
        # train mode
        self.model.train()
        return vl, va, vap

    def evaluate_gfsl(self, fsl_loader, gfsl_loader, gfsl_dataset):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.eval()
        record = np.zeros((args.num_eval_episodes, 5)) # loss and acc
        label_unseen_query = torch.arange(min(args.eval_way, self.valset.num_class)).repeat(args.eval_query).long()
        if torch.cuda.is_available():
            label_unseen_query = label_unseen_query.cuda()
        print('{} best epoch {}, best val acc={:.4f} + {:.4f}'.format(
            args.test_mode,
            self.trlog['max_acc_epoch'],
            self.trlog['max_acc'],
            self.trlog['max_acc_interval']))
        with torch.no_grad():
            for i, batch in tqdm(enumerate(zip(gfsl_loader, fsl_loader), 1)):
                if torch.cuda.is_available():
                    data_seen, data_unseen, seen_label, unseen_label = batch[0][0].cuda(), batch[1][0].cuda(), batch[0][1].cuda(), batch[1][1].cuda()
                else:
                    data_seen, data_unseen, seen_label, unseen_label = batch[0][0], batch[1][0], batch[0][1], batch[1][1]

                p2 = args.eval_shot * args.eval_way
                data_unseen_shot, data_unseen_query = data_unseen[:p2], data_unseen[p2:]
                label_unseen_shot, _ = unseen_label[:p2], unseen_label[p2:]
                whole_query = torch.cat([data_seen, data_unseen_query], 0)
                whole_label = torch.cat([seen_label, label_unseen_query + gfsl_dataset.num_class])              
                logits_s, logits_u = self.model.forward_generalized(data_unseen_shot, whole_query)
                # compute un-biased accuracy
                new_logits = torch.cat([logits_s, logits_u], 1)
                record[i-1, 0] = F.cross_entropy(new_logits, whole_label).item()
                record[i-1, 1] = count_acc(new_logits, whole_label)
                # compute harmonic mean
                HM_nobias, SA_nobias, UA_nobias = count_acc_harmonic_low_shot_joint(torch.cat([logits_s, logits_u], 1), 
                                                                                    whole_label, seen_label.shape[0])
                record[i-1, 2:] = np.array([HM_nobias, SA_nobias, UA_nobias])
                del logits_s, logits_u, new_logits
                torch.cuda.empty_cache()
                
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,2])
        
        # train mode
        self.model.train()
        return vl, va, vap

    def evaluate_test(self):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc.pth'))['params'])
        self.model.eval()
        
        if args.test_mode == 'FSL':
            record = np.zeros((10000, 2)) # loss and acc
            label = torch.arange(args.eval_way).repeat(args.eval_query).type(torch.LongTensor)
            if torch.cuda.is_available():
                label = label.cuda()
            with torch.no_grad():
                for i, batch in enumerate(self.test_fsl_loader, 1):
                    if torch.cuda.is_available():
                        data, _ = [_.cuda() for _ in batch]
                    else:
                        data = batch[0]
                        
                    p = args.eval_shot * args.eval_way
                    data_shot, data_query = data[:p], data[p:] 
                    logits = self.model.forward_fsl(data_shot, data_query)
                    loss = F.cross_entropy(logits, label)
                    acc = count_acc(logits, label)
                    record[i-1, 0] = loss.item()
                    record[i-1, 1] = acc
            assert(i == record.shape[0])
            vl, _ = compute_confidence_interval(record[:,0])
            va, vap = compute_confidence_interval(record[:,1])
        
            self.trlog['test_acc'] = va
            self.trlog['test_acc_interval'] = vap
            self.trlog['test_loss'] = vl
                    
            print('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                    self.trlog['max_acc_epoch'],
                        self.trlog['max_acc'],
                        self.trlog['max_acc_interval']))
            print('Test acc={:.4f} + {:.4f}\n'.format(
                    self.trlog['test_acc'],
                        self.trlog['test_acc_interval']))
        
        else:
            record = np.zeros((10000, 5)) # loss and acc
            label_unseen_query = torch.arange(min(args.eval_way, self.valset.num_class)).repeat(args.eval_query).long()
            if torch.cuda.is_available():
                label_unseen_query = label_unseen_query.cuda()
            with torch.no_grad():        
                for i, batch in tqdm(enumerate(zip(self.test_gfsl_loader, self.test_fsl_loader), 1)):
                    if torch.cuda.is_available():
                        data_seen, data_unseen, seen_label, unseen_label = batch[0][0].cuda(), batch[1][0].cuda(), batch[0][1].cuda(), batch[1][1].cuda()
                    else:
                        data_seen, data_unseen, seen_label, unseen_label = batch[0][0], batch[1][0], batch[0][1], batch[1][1]
                    p2 = args.eval_shot * args.eval_way
                    data_unseen_shot, data_unseen_query = data_unseen[:p2], data_unseen[p2:]
                    label_unseen_shot, _ = unseen_label[:p2], unseen_label[p2:]
                    whole_query = torch.cat([data_seen, data_unseen_query], 0)
                    whole_label = torch.cat([seen_label, label_unseen_query + self.traintestset.num_class])      
                    logits_s, logits_u = self.model.forward_generalized(data_unseen_shot, whole_query)
                    # compute un-biased accuracy
                    new_logits = torch.cat([logits_s, logits_u], 1)
                    record[i-1, 0] = F.cross_entropy(new_logits, whole_label).item()
                    record[i-1, 1] = count_acc(new_logits, whole_label)
                    # compute harmonic mean
                    HM_nobias, SA_nobias, UA_nobias = count_acc_harmonic_low_shot_joint(torch.cat([logits_s, logits_u], 1), 
                                                                                                    whole_label, seen_label.shape[0])
                    record[i-1, 2:] = np.array([HM_nobias, SA_nobias, UA_nobias])
                    del logits_s, logits_u, new_logits
                    torch.cuda.empty_cache()
                          
            m_list = []
            p_list = []
            for i in range(5):
                m1, p1 = compute_confidence_interval(record[:,i])
                m_list.append(m1)
                p_list.append(p1)
            
            self.trlog['test_loss'] = m_list[0]
            self.trlog['test_acc'] = m_list[1]
            self.trlog['test_acc_interval'] = p_list[1]
            self.trlog['test_HM_acc'] = m_list[2]
            self.trlog['test_HM_acc_interval'] = p_list[2]
            self.trlog['test_HMSeen_acc'] = m_list[3]
            self.trlog['test_HMSeen_acc_interval'] = p_list[3]            
            self.trlog['test_HMUnseen_acc'] = m_list[4]
            self.trlog['test_HMUnseen_acc_interval'] = p_list[4]                     
                        
            print('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
            print('Test HM acc={:.4f} + {:.4f}\n'.format(
                    self.trlog['test_HM_acc'],
                    self.trlog['test_HM_acc_interval']))               
            print('GFSL {}-way Acc w/o Bias {:.5f} + {:.5f}'.format(args.eval_way, m_list[1], p_list[1]))
            print('GFSL {}-way HM  w/o Bias {:.5f} + {:.5f}'.format(args.eval_way, m_list[2], p_list[2]))
            print('GFSL {}-way HMSeen  w/o Bias {:.5f} + {:.5f}'.format(args.eval_way, m_list[3], p_list[3]))
            print('GFSL {}-way HMUnseen  w/o Bias {:.5f} + {:.5f}'.format(args.eval_way, m_list[4], p_list[4]))          

    def final_record(self):
        # save the best performance in a txt file
        if self.args.test_mode == 'FSL':
            with open(osp.join(self.args.save_path, '{}-{}+{}.txt'.format('FSL', self.trlog['test_acc'], self.trlog['test_acc_interval'])), 'w') as f:
                f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                        self.trlog['max_acc_epoch'],
                        self.trlog['max_acc'],
                        self.trlog['max_acc_interval']))
                f.write('Test acc={:.4f} + {:.4f}\n'.format(
                        self.trlog['test_acc'],
                        self.trlog['test_acc_interval']))                        
        else:
            with open(osp.join(self.args.save_path, '{}-{}+{}.txt'.format('GFSL', self.trlog['test_HM_acc'], self.trlog['test_HM_acc_interval'])), 'w') as f:
                f.write('best epoch {}, best val HM acc={:.4f} + {:.4f}\n'.format(
                        self.trlog['max_acc_epoch'],
                        self.trlog['max_acc'],
                        self.trlog['max_acc_interval'])) 
                f.write('GFSL {}-way Acc w/o Bias {:.5f} + {:.5f}\n'.format(self.args.eval_way, self.trlog['test_acc'], self.trlog['test_acc_interval']))
                f.write('GFSL {}-way HM  w/o Bias {:.5f} + {:.5f}\n'.format(self.args.eval_way, self.trlog['test_HM_acc'], self.trlog['test_HM_acc_interval']))
                f.write('GFSL {}-way HMSeen  w/o Bias {:.5f} + {:.5f}\n'.format(self.args.eval_way, self.trlog['test_HMSeen_acc'], self.trlog['test_HMSeen_acc_interval']))
                f.write('GFSL {}-way HMUnseen  w/o Bias {:.5f} + {:.5f}\n'.format(self.args.eval_way, self.trlog['test_HMUnseen_acc'], self.trlog['test_HMUnseen_acc_interval']))                  