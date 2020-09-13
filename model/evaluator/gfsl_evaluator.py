import time
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F

from model.evaluator.base import Evaluator
from model.evaluator.helpers import (
    get_dataloader, prepare_model,
)
from model.utils import (
    pprint, ensure_path,
    Averager, Timer, count_acc, one_hot,
    compute_confidence_interval, count_acc_harmonic_low_shot_joint, count_acc_harmonic_MAP, count_delta_value, Compute_AUSUC
)
from tqdm import tqdm
import pickle

class GFSLEvaluator(Evaluator):
    def __init__(self, args):
        super().__init__(args)

        self.trainset, self.valset, self.trainvalset, self.testset, self.traintestset, \
            self.train_fsl_loader, self.train_gfsl_loader, self.train_proto_loader, self.val_fsl_loader, self.val_gfsl_loader, self.test_fsl_loader, self.test_gfsl_loader = get_dataloader(args)
        self.model = prepare_model(args)
        self.model.eval()
        # sample training class prototpyes if needed
        if args.model_class in ['ProtoNet']:
            seen_proto = []
            with torch.no_grad():
                for batch in tqdm(self.train_proto_loader):
                    if torch.cuda.is_available():
                        data, label = batch[0].cuda(), batch[1].cuda()
                    else:
                        data, label = batch[0], batch[1]
                    seen_proto.append(self.model.encoder(data).mean(dim=0))
            self.model.seen_proto = torch.stack(seen_proto)            
            
    def get_calibration(self):
        # tune calibration factors on the validation set
        args = self.args
        label_unseen_query = torch.arange(min(self.args.eval_way, self.valset.num_class)).repeat(self.args.eval_query).long()
        if torch.cuda.is_available():
            label_unseen_query = label_unseen_query.cuda()
    
        # sample 500 tasks and get the range of the bias
        range_list = np.zeros((500, ))
        for i, batch in tqdm(enumerate(zip(self.val_gfsl_loader, self.val_fsl_loader), 1)):
            if torch.cuda.is_available():
                data_seen, data_unseen, seen_label, unseen_label = batch[0][0].cuda(), batch[1][0].cuda(), batch[0][1].cuda(), batch[1][1].cuda()
            else:
                data_seen, data_unseen, seen_label, unseen_label = batch[0][0], batch[1][0], batch[0][1], batch[1][1]
            
            p2 = self.args.eval_shot * min(self.args.eval_way, self.valset.num_class)
            data_unseen_shot, data_unseen_query = data_unseen[:p2], data_unseen[p2:]
            label_unseen_shot, _ = unseen_label[:p2], unseen_label[p2:]  
            whole_query = torch.cat([data_seen, data_unseen_query], 0)
            whole_label = torch.cat([seen_label, label_unseen_query + self.trainset.num_class])            
            
            if self.args.model_class in ['CLS', 'Castle', 'ACastle']:
                with torch.no_grad():
                    logits_s, logits_u = self.model.forward_generalized(data_unseen_shot, whole_query)
            elif self.args.model_class in ['ProtoNet']:
                with torch.no_grad():
                    logits_s, logits_u = self.model.forward_generalized(data_unseen_shot, whole_query, self.model.seen_proto)    
            else:
                raise ValueError('No Such Model')
                    
            range_list[i-1] = torch.max(torch.max(logits_s, dim=1)[0] - torch.max(logits_u, dim=1)[0]).item()
            del logits_s, logits_u
            torch.cuda.empty_cache()
            
        possible_range = np.mean(range_list)
        bias_candidate = np.linspace(-possible_range, possible_range, 30)
        if 'acc' in self.criteria or 'hmeanacc' in self.criteria or 'delta' in self.criteria:
            tune_params_acc = np.zeros((500, 30))
        if 'hmeanmap' in self.criteria:
            tune_params_map = np.zeros((500, 30))
        else:
            raise ValueError ( 'Criteria Error' )
        
        for i, batch in tqdm(enumerate(zip(self.val_gfsl_loader, self.val_fsl_loader), 1)):
            if torch.cuda.is_available():
                data_seen, data_unseen, seen_label, unseen_label = batch[0][0].cuda(), batch[1][0].cuda(), batch[0][1].cuda(), batch[1][1].cuda()
            else:
                data_seen, data_unseen, seen_label, unseen_label = batch[0][0], batch[1][0], batch[0][1], batch[1][1]
            
            p2 = self.args.eval_shot * min(self.args.eval_way, self.valset.num_class)
            data_unseen_shot, data_unseen_query = data_unseen[:p2], data_unseen[p2:]
            label_unseen_shot, _ = unseen_label[:p2], unseen_label[p2:]
            whole_query = torch.cat([data_seen, data_unseen_query], 0)
            whole_label = torch.cat([seen_label, label_unseen_query + self.trainset.num_class])            
            
            if self.args.model_class in ['CLS', 'Castle', 'ACastle']:
                with torch.no_grad():
                    logits_s, logits_u = self.model.forward_generalized(data_unseen_shot, whole_query)
            elif self.args.model_class in ['ProtoNet']:
                with torch.no_grad():
                    logits_s, logits_u = self.model.forward_generalized(data_unseen_shot, whole_query, self.model.seen_proto)              
            else:
                raise ValueError('No Such Model')
            
            for pr_index, pr in enumerate(bias_candidate):
                if 'acc' in self.criteria or 'hmeanacc' in self.criteria or 'delta' in self.criteria:
                    tune_params_acc[i-1, pr_index], _, _ = count_acc_harmonic_low_shot_joint(torch.cat([logits_s - pr, logits_u], 1), 
                                                                                             whole_label, 
                                                                                             seen_label.shape[0])
                if 'hmeanmap' in self.criteria:
                    tune_params_map[i-1, pr_index], _, _ = count_acc_harmonic_MAP(torch.cat([logits_s - pr, logits_u], 1), 
                                                                                  whole_label, 
                                                                                  seen_label.shape[0], 
                                                                                  'macro') # average_mode, we fix macro here  
    
            del logits_s, logits_u
            torch.cuda.empty_cache()
                    
        # get best bias
        if 'acc' in self.criteria or 'hmeanacc' in self.criteria or 'delta' in self.criteria:
            tune_params_acc = np.mean(tune_params_acc, 0)
            best_bias_acc = bias_candidate[np.argmax(tune_params_acc)]    
            self.best_bias_acc = best_bias_acc
            print('Best Acc Bias {}'.format(self.best_bias_acc)) 
            
        if 'hmeanmap' in self.criteria:
            tune_params_map = np.mean(tune_params_map, 0)
            best_bias_map = bias_candidate[np.argmax(tune_params_map)]    
            self.best_bias_map = best_bias_map   
            print('Best MAP Bias {}'.format(self.best_bias_map)) 
    
    def evaluate_gfsl(self):
        args = self.args        
        label_unseen_query = torch.arange(args.eval_way).repeat(args.eval_query).long()
        if torch.cuda.is_available():
            label_unseen_query = label_unseen_query.cuda()
            
        generalized_few_shot_acc = np.zeros((args.num_eval_episodes, 2))
        generalized_few_shot_delta = np.zeros((args.num_eval_episodes, 4))
        generalized_few_shot_hmeanacc = np.zeros((args.num_eval_episodes, 6))
        generalized_few_shot_hmeanmap = np.zeros((args.num_eval_episodes, 6))
        generalized_few_shot_ausuc = np.zeros((args.num_eval_episodes, 1))
        AUC_record = []
        
        for i, batch in tqdm(enumerate(zip(self.test_gfsl_loader, self.test_fsl_loader), 1)):
            if torch.cuda.is_available():
                data_seen, data_unseen, seen_label, unseen_label = batch[0][0].cuda(), batch[1][0].cuda(), batch[0][1].cuda(), batch[1][1].cuda()
            else:
                data_seen, data_unseen, seen_label, unseen_label = batch[0][0], batch[1][0], batch[0][1], batch[1][1]
            p2 = args.eval_shot * args.eval_way
            
            data_unseen_shot, data_unseen_query = data_unseen[:p2], data_unseen[p2:]
            label_unseen_shot, _ = unseen_label[:p2], unseen_label[p2:]
            whole_query = torch.cat([data_seen, data_unseen_query], 0)
            whole_label = torch.cat([seen_label, label_unseen_query + self.trainset.num_class])            
            if args.model_class in ['CLS', 'Castle', 'ACastle']:
                with torch.no_grad():
                    logits_s, logits_u = self.model.forward_generalized(data_unseen_shot, whole_query)
            elif args.model_class in ['ProtoNet']:
                with torch.no_grad():
                    logits_s, logits_u = self.model.forward_generalized(data_unseen_shot, whole_query, self.model.seen_proto)        
            # compute un-biased accuracy
            new_logits = torch.cat([logits_s, logits_u], 1)
            if 'acc' in self.criteria or 'hmeanacc' in self.criteria or 'delta' in self.criteria:
                new_logits_acc_biased = torch.cat([logits_s - self.best_bias_acc, logits_u], 1)
            if 'hmeanmap' in self.criteria:
                new_logits_map_biased = torch.cat([logits_s - self.best_bias_map, logits_u], 1)
            # Criterion: Acc
            if 'acc' in self.criteria:
                generalized_few_shot_acc[i-1, 0] = count_acc(new_logits, whole_label)
                # compute biased accuracy
                generalized_few_shot_acc[i-1, 1] = count_acc(new_logits_acc_biased, whole_label)
            
            if 'delta' in self.criteria:
                # compute delta value for un-biased logits
                unbiased_detla1, unbiased_detla2 = count_delta_value(new_logits, 
                                                                     whole_label, 
                                                                     seen_label.shape[0], 
                                                                     self.trainset.num_class)
                # compute delta value
                biased_detla1, biased_detla2 = count_delta_value(new_logits_acc_biased, 
                                                                 whole_label, 
                                                                 seen_label.shape[0], 
                                                                 self.trainset.num_class)
                generalized_few_shot_delta[i-1, :] = np.array([unbiased_detla1, 
                                                               unbiased_detla2, 
                                                               biased_detla1, 
                                                               biased_detla2])
            
            if 'hmeanacc' in self.criteria:
                # compute harmonic mean
                HM_nobias, SA_nobias, UA_nobias = count_acc_harmonic_low_shot_joint(new_logits, 
                                                                                    whole_label, 
                                                                                    seen_label.shape[0])
                HM, SA, UA = count_acc_harmonic_low_shot_joint(new_logits_acc_biased, 
                                                               whole_label, 
                                                               seen_label.shape[0])        
                generalized_few_shot_hmeanacc[i-1, :] = np.array([HM_nobias, SA_nobias, UA_nobias, HM, SA, UA])
                
            if 'hmeanmap' in self.criteria:
                # compute harmonic mean
                HM_nobias, SA_nobias, UA_nobias = count_acc_harmonic_MAP(new_logits, 
                                                                         whole_label, 
                                                                         seen_label.shape[0], 
                                                                         'macro')
                HM, SA, UA = count_acc_harmonic_MAP(new_logits_map_biased, 
                                                    whole_label, 
                                                    seen_label.shape[0], 
                                                    'macro')        
                generalized_few_shot_hmeanmap[i-1, :] = np.array([HM_nobias, SA_nobias, UA_nobias, HM, SA, UA])
    
            
            if 'ausuc' in self.criteria:
                # compute AUSUC
                generalized_few_shot_ausuc[i-1, 0], temp_auc_record = Compute_AUSUC(logits_s.detach().cpu().numpy(), 
                                                                   logits_u.detach().cpu().numpy(), 
                                                                   whole_label.cpu().numpy(), 
                                                                   np.arange(self.trainset.num_class), 
                                                                   self.trainset.num_class + np.arange(args.eval_way))
                AUC_record.append(temp_auc_record)
            
            del logits_s, logits_u, new_logits
            torch.cuda.empty_cache()
        
        self.AUC_record = AUC_record
        print('-'.join([args.model_class, args.model_path]))
        if 'acc' in self.criteria:
            self.trlog['acc_mean'], self.trlog['acc_interval'] = compute_confidence_interval(generalized_few_shot_acc[:, 0])
            self.trlog['acc_biased_mean'], self.trlog['acc_biased_interval'] = compute_confidence_interval(generalized_few_shot_acc[:, 1])
            print('GFSL {}-way Acc w/o Bias {:.5f} + {:.5f}'.format(args.eval_way, self.trlog['acc_mean'], self.trlog['acc_interval']))
            print('GFSL {}-way Acc w/  Bias {:.5f} + {:.5f}'.format(args.eval_way, self.trlog['acc_biased_mean'], self.trlog['acc_biased_interval']))        
            
        if 'delta' in self.criteria:
            self.trlog['detla1_mean'], self.trlog['detla1_interval'] = compute_confidence_interval(generalized_few_shot_delta[:, 0])
            self.trlog['detla2_mean'], self.trlog['detla2_interval'] = compute_confidence_interval(generalized_few_shot_delta[:, 1])
            self.trlog['detla1_biased_mean'], self.trlog['detla1_biased_interval'] = compute_confidence_interval(generalized_few_shot_delta[:, 2])
            self.trlog['detla2_biased_mean'], self.trlog['detla2_biased_interval'] = compute_confidence_interval(generalized_few_shot_delta[:, 3])
            print('GFSL {}-way Detla1 w/o Bias {:.5f} + {:.5f}'.format(args.eval_way, self.trlog['detla1_mean'], self.trlog['detla1_interval']))
            print('GFSL {}-way Detla1 w/  Bias {:.5f} + {:.5f}'.format(args.eval_way, self.trlog['detla1_biased_mean'], self.trlog['detla1_biased_interval']))  
            print('GFSL {}-way Detla2 w/o Bias {:.5f} + {:.5f}'.format(args.eval_way, self.trlog['detla2_mean'], self.trlog['detla2_interval']))
            print('GFSL {}-way Detla2 w/  Bias {:.5f} + {:.5f}'.format(args.eval_way, self.trlog['detla2_biased_mean'], self.trlog['detla2_biased_interval']))
        
        if 'hmeanacc' in self.criteria:
            self.trlog['HM_mean'], self.trlog['HM_interval'] = compute_confidence_interval(generalized_few_shot_hmeanacc[:, 0])
            self.trlog['S2All_mean'], self.trlog['S2All_interval'] = compute_confidence_interval(generalized_few_shot_hmeanacc[:, 1])
            self.trlog['U2All_mean'], self.trlog['U2All_interval'] = compute_confidence_interval(generalized_few_shot_hmeanacc[:, 2])
            self.trlog['HM_biased_mean'], self.trlog['HM_biased_nterval'] = compute_confidence_interval(generalized_few_shot_hmeanacc[:, 3])
            self.trlog['S2All_biased_mean'], self.trlog['S2All_biased_interval'] = compute_confidence_interval(generalized_few_shot_hmeanacc[:, 4])
            self.trlog['U2All_biased_mean'], self.trlog['U2All_biased_interval'] = compute_confidence_interval(generalized_few_shot_hmeanacc[:, 5])
            print('GFSL {}-way HM_mean w/o Bias {:.5f} + {:.5f}'.format(args.eval_way, self.trlog['HM_mean'], self.trlog['HM_interval']))
            print('GFSL {}-way HM_mean w/  Bias {:.5f} + {:.5f}'.format(args.eval_way, self.trlog['HM_biased_mean'], self.trlog['HM_biased_nterval']))  
            print('GFSL {}-way S2All_mean w/o Bias {:.5f} + {:.5f}'.format(args.eval_way, self.trlog['S2All_mean'], self.trlog['S2All_interval']))
            print('GFSL {}-way S2All_mean w/  Bias {:.5f} + {:.5f}'.format(args.eval_way, self.trlog['S2All_biased_mean'], self.trlog['S2All_biased_interval']))
            print('GFSL {}-way U2All_mean w/o Bias {:.5f} + {:.5f}'.format(args.eval_way, self.trlog['U2All_mean'], self.trlog['U2All_interval']))
            print('GFSL {}-way U2All_mean w/  Bias {:.5f} + {:.5f}'.format(args.eval_way, self.trlog['U2All_biased_mean'], self.trlog['U2All_biased_interval']))            
            
        if 'hmeanmap' in self.criteria:
            self.trlog['HM_map_mean'], self.trlog['HM_map_interval'] = compute_confidence_interval(generalized_few_shot_hmeanmap[:, 0])
            self.trlog['S2All_map_mean'], self.trlog['S2All_map_interval'] = compute_confidence_interval(generalized_few_shot_hmeanmap[:, 1])
            self.trlog['U2All_map_mean'], self.trlog['U2All_map_interval'] = compute_confidence_interval(generalized_few_shot_hmeanmap[:, 2])
            self.trlog['HM_map_biased_mean'], self.trlog['HM_map_biased_nterval'] = compute_confidence_interval(generalized_few_shot_hmeanmap[:, 3])
            self.trlog['S2All_map_biased_mean'], self.trlog['S2All_map_biased_interval'] = compute_confidence_interval(generalized_few_shot_hmeanmap[:, 4])
            self.trlog['U2All_map_biased_mean'], self.trlog['U2All_map_biased_interval'] = compute_confidence_interval(generalized_few_shot_hmeanmap[:, 5])
            print('GFSL {}-way HM_map_mean w/o Bias {:.5f} + {:.5f}'.format(args.eval_way, self.trlog['HM_map_mean'], self.trlog['HM_map_interval']))
            print('GFSL {}-way HM_map_mean w/  Bias {:.5f} + {:.5f}'.format(args.eval_way, self.trlog['HM_map_biased_mean'], self.trlog['HM_map_biased_nterval']))  
            print('GFSL {}-way S2All_map_mean w/o Bias {:.5f} + {:.5f}'.format(args.eval_way, self.trlog['S2All_map_mean'], self.trlog['S2All_map_interval']))
            print('GFSL {}-way S2All_map_mean w/  Bias {:.5f} + {:.5f}'.format(args.eval_way, self.trlog['S2All_map_biased_mean'], self.trlog['S2All_map_biased_interval']))
            print('GFSL {}-way U2All_map_mean w/o Bias {:.5f} + {:.5f}'.format(args.eval_way, self.trlog['U2All_map_mean'], self.trlog['U2All_map_interval']))
            print('GFSL {}-way U2All_map_mean w/  Bias {:.5f} + {:.5f}'.format(args.eval_way, self.trlog['U2All_map_biased_mean'], self.trlog['U2All_map_biased_interval']))        
        
        if 'ausuc' in self.criteria:
            self.trlog['AUSUC_mean'], self.trlog['AUSUC_interval'] = compute_confidence_interval(generalized_few_shot_ausuc[:, 0])             
            print('GFSL {}-way AUSUC {:.5f} + {:.5f}'.format(args.eval_way, self.trlog['AUSUC_mean'], self.trlog['AUSUC_interval']))
            
    def final_record(self):
        args = self.args
        # save the best performance in a txt file
        save_path = osp.join(*args.model_path.split('/')[:-1])
        if 'ausuc' in self.criteria:
            with open(osp.join(save_path, '{}-{}-{}.dat'.format(args.eval_way, args.eval_shot, args.model_class)), 'wb') as f:
                    pickle.dump({'AUC_record': self.AUC_record,
                                 'AUSUC_mean': self.trlog['AUSUC_mean'],
                                 'AUSUC_std': self.trlog['AUSUC_interval']}, f)           
                    
        if 'hmeanacc' in self.criteria:
            with open(osp.join(save_path, '{}-{}-way: {}+{}.txt'.format('GFSL', args.eval_way, self.trlog['HM_mean'], self.trlog['HM_interval'])), 'w') as f:
                if 'acc' in self.criteria:
                    f.write('GFSL {}-way Acc w/o Bias {:.5f} + {:.5f}\n'.format(args.eval_way, self.trlog['acc_mean'], self.trlog['acc_interval']))
                    f.write('GFSL {}-way Acc w/  Bias {:.5f} + {:.5f}\n'.format(args.eval_way, self.trlog['acc_biased_mean'], self.trlog['acc_biased_interval']))       
                if 'delta' in self.criteria:
                    f.write('GFSL {}-way Detla1 w/o Bias {:.5f} + {:.5f}\n'.format(args.eval_way, self.trlog['detla1_mean'], self.trlog['detla1_interval']))
                    f.write('GFSL {}-way Detla1 w/  Bias {:.5f} + {:.5f}\n'.format(args.eval_way, self.trlog['detla1_biased_mean'], self.trlog['detla1_biased_interval']))  
                    f.write('GFSL {}-way Detla2 w/o Bias {:.5f} + {:.5f}\n'.format(args.eval_way, self.trlog['detla2_mean'], self.trlog['detla2_interval']))
                    f.write('GFSL {}-way Detla2 w/  Bias {:.5f} + {:.5f}\n'.format(args.eval_way, self.trlog['detla2_biased_mean'], self.trlog['detla2_biased_interval']))                
                if 'hmeanacc' in self.criteria:
                    f.write('GFSL {}-way HM_mean w/o Bias {:.5f} + {:.5f}\n'.format(args.eval_way, self.trlog['HM_mean'], self.trlog['HM_interval']))
                    f.write('GFSL {}-way HM_mean w/  Bias {:.5f} + {:.5f}\n'.format(args.eval_way, self.trlog['HM_biased_mean'], self.trlog['HM_biased_nterval']))  
                    f.write('GFSL {}-way S2All_mean w/o Bias {:.5f} + {:.5f}\n'.format(args.eval_way, self.trlog['S2All_mean'], self.trlog['S2All_interval']))
                    f.write('GFSL {}-way S2All_mean w/  Bias {:.5f} + {:.5f}\n'.format(args.eval_way, self.trlog['S2All_biased_mean'], self.trlog['S2All_biased_interval']))
                    f.write('GFSL {}-way U2All_mean w/o Bias {:.5f} + {:.5f}\n'.format(args.eval_way, self.trlog['U2All_mean'], self.trlog['U2All_interval']))
                    f.write('GFSL {}-way U2All_mean w/  Bias {:.5f} + {:.5f}\n'.format(args.eval_way, self.trlog['U2All_biased_mean'], self.trlog['U2All_biased_interval']))   
                if 'hmeanmap' in self.criteria:
                    f.write('GFSL {}-way HM_map_mean w/o Bias {:.5f} + {:.5f}\n'.format(args.eval_way, self.trlog['HM_map_mean'], self.trlog['HM_map_interval']))
                    f.write('GFSL {}-way HM_map_mean w/  Bias {:.5f} + {:.5f}\n'.format(args.eval_way, self.trlog['HM_map_biased_mean'], self.trlog['HM_map_biased_nterval']))  
                    f.write('GFSL {}-way S2All_map_mean w/o Bias {:.5f} + {:.5f}\n'.format(args.eval_way, self.trlog['S2All_map_mean'], self.trlog['S2All_map_interval']))
                    f.write('GFSL {}-way S2All_map_mean w/  Bias {:.5f} + {:.5f}\n'.format(args.eval_way, self.trlog['S2All_map_biased_mean'], self.trlog['S2All_map_biased_interval']))
                    f.write('GFSL {}-way U2All_map_mean w/o Bias {:.5f} + {:.5f}\n'.format(args.eval_way, self.trlog['U2All_map_mean'], self.trlog['U2All_map_interval']))
                    f.write('GFSL {}-way U2All_map_mean w/  Bias {:.5f} + {:.5f}\n'.format(args.eval_way, self.trlog['U2All_map_biased_mean'], self.trlog['U2All_map_biased_interval']))        
                if 'ausuc' in self.criteria:
                    f.write('GFSL {}-way AUSUC {:.5f} + {:.5f}\n'.format(args.eval_way, self.trlog['AUSUC_mean'], self.trlog['AUSUC_interval']))
            