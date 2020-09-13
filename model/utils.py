import os
import shutil
import time
import pprint
import torch
import argparse
import numpy as np

## ------------------------ Basic Functions ------------------------ 
def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
    if indices.is_cuda:
        encoded_indicies = encoded_indicies.cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)

    return encoded_indicies

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def ensure_path(dir_path, scripts_to_save=None):
    if os.path.exists(dir_path):
        if input('{} exists, remove? ([y]/n)'.format(dir_path)) != 'n':
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)
    else:
        os.mkdir(dir_path)

    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, 'scripts')
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for src_file in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(src_file))
            print('copy {} to {}'.format(src_file, dst_file))
            if os.path.isdir(src_file):
                shutil.copytree(src_file, dst_file)
            else:
                shutil.copyfile(src_file, dst_file)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def count_acc2(logits, label, mask1, mask2):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        temp = (pred == label).type(torch.cuda.FloatTensor)
        acc = (torch.mean(temp.masked_select(mask1)) + torch.mean(temp.masked_select(mask2))) / 2
    else:
        temp = (pred == label).type(torch.FloatTensor)
        acc = (torch.mean(temp.masked_select(mask1)) + torch.mean(temp.masked_select(mask2))) / 2
    return acc.item()

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm

def top1accuracy(output, target):
    _, pred = output.max(dim=1)
    pred = pred.view(-1)
    target = target.view(-1)
    accuracy = 100 * pred.eq(target).float().mean()
    return accuracy

## ------------------------ GFSL Measures ------------------------ 
def count_acc_harmonic(logits, label, th):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        result = (pred == label).type(torch.cuda.FloatTensor)
    else:
        result =  (pred == label).type(torch.FloatTensor)
    seen_acc = result[:th].mean().item()
    unseen_acc = result[th:].mean().item()
    return 2 * (seen_acc * unseen_acc) / (seen_acc + unseen_acc + 1e-12)


# the method to count harmonic mean in low-shot learning paper
def count_acc_harmonic_low_shot(logits, label, th, nKbase):
    seen_acc = top1accuracy(logits[:th,:nKbase], label[:th])        
    unseen_acc = top1accuracy(logits[th:,nKbase:], (label[th:]-nKbase))
    return 2 * (seen_acc * unseen_acc) / (seen_acc + unseen_acc + 1e-12), seen_acc, unseen_acc

# based on the seen-joint and unseen_joint performnace
def count_acc_harmonic_low_shot_joint(logits, label, th):
    seen_acc = top1accuracy(logits[:th, :], label[:th])        
    unseen_acc = top1accuracy(logits[th:, :], label[th:])
    return 2 * (seen_acc * unseen_acc) / (seen_acc + unseen_acc + 1e-12), seen_acc, unseen_acc

def count_delta_value(logits, label, th, nKbase):
    seen_acc = top1accuracy(logits[:th,:nKbase], label[:th])        
    unseen_acc = top1accuracy(logits[th:,nKbase:], (label[th:]-nKbase))
    joint_acc = top1accuracy(logits, label)
    delta1 = 0.5 * (seen_acc + unseen_acc - 2 * joint_acc)
    seen_acc_joint = top1accuracy(logits[:th, :], label[:th])        
    unseen_acc_joint = top1accuracy(logits[th:, :], label[th:])    
    delta2 = 0.5 * ((seen_acc - seen_acc_joint) + (unseen_acc - unseen_acc_joint))
    return delta1, delta2

from sklearn.metrics import average_precision_score
def compute_map(output, target, average_mode):
    num_class = output.shape[1]
    target = one_hot(target, num_class).cpu().numpy()
    output = output.detach().cpu().numpy()
    return average_precision_score(target, output, average = average_mode)

def compute_weight_map(output, target, th2=64):
    num_class = output.shape[1]
    target = one_hot(target, num_class).cpu().numpy()
    output = output.cpu().numpy()
    map_list1 = []
    map_list2 = []
    selected_mask1 = np.where(np.sum(target[:,:th2], 0) > 0)[0]
    selected_mask2 = np.where(np.sum(target[:,th2:], 0) > 0)[0] + th2
    for i in selected_mask1:
        map_list1.append(average_precision_score(target[:,i], output[:,i]))
    for i in selected_mask2:
        map_list2.append(average_precision_score(target[:,i], output[:,i]))        
    map_seen = np.array(map_list1)
    map_unseen = np.array(map_list2)
    return np.mean(map_seen), np.mean(map_unseen)

# based on the seen-joint and unseen_joint performnace based on MAP
# change recall = tps / tps[-1] in sklearn/metrics/ranking.py to recall = np.ones(tps.size) if tps[-1] == 0 else tps / tps[-1]
def count_acc_harmonic_MAP(logits, label, th, average_mode = 'macro'):
    if average_mode == 'weighted':
        seen_map, unseen_map = compute_weight_map(logits, label)
    else:
        seen_map = compute_map(logits[:th, :], label[:th], average_mode)        
        unseen_map = compute_map(logits[th:, :], label[th:], average_mode)
    return 2 * (seen_map * unseen_map) / (seen_map + unseen_map + 1e-12), seen_map, unseen_map

def AUC_eval_class_count(Ypred_S, Ypred_U, label_S, label_U, Ytrue):
    # get number counts for AUC evaluation
    L_S = label_S.shape[0]
    L_U = label_U.shape[0]
    class_count_S = np.zeros((L_S, 1))
    class_count_U = np.zeros((L_U, 1))
    class_correct_S = np.zeros((L_S, 1))
    class_correct_U = np.zeros((L_U, 1))
    
    class_count_S = [sum(Ytrue == label_S[i]) for i in range(L_S)]
    class_correct_S = [sum((Ytrue == label_S[i]) & (Ypred_S == label_S[i])) for i in range(L_S)]
    class_count_U = [sum(Ytrue == label_U[i]) for i in range(L_U)]
    class_correct_U = [sum((Ytrue == label_U[i]) & (Ypred_U == label_U[i])) for i in range(L_U)]    
    
    class_count_S = np.array(class_count_S)
    class_correct_S = np.array(class_correct_S)
    class_count_U = np.array(class_count_U)
    class_correct_U = np.array(class_correct_U)

    class_count_S[class_count_S == 0] = 10 ^ 10;
    class_count_U[class_count_U == 0] = 10 ^ 10;

    return class_correct_S, class_correct_U, class_count_S, class_count_U

def Compute_HM(acc):
    HM = 2 * acc[0] * acc[1] / (acc[0] + acc[1] + 1e-12);
    return HM

def Compute_AUSUC(score_S, score_U, Y, label_S, label_U):
    Y = Y.reshape(-1)
    label_S = label_S.reshape(-1)
    label_U = label_U.reshape(-1)    
    AUC_record = np.zeros((Y.shape[0] + 1, 2))
    label_S = np.unique(label_S)
    label_U = np.unique(label_U)
    L_S = label_S.shape[0]
    L_U = label_U.shape[0]
    num_inst = score_S.shape[0]

    # effective bias searching
    loc_S = np.argmax(score_S, axis=1)
    max_S = score_S[np.arange(num_inst), loc_S]
    Ypred_S = label_S[loc_S]
    loc_U = np.argmax(score_U, axis=1)
    max_U = score_U[np.arange(num_inst), loc_U]    
    Ypred_U = label_U[loc_U]
    class_correct_S, class_correct_U, class_count_S, class_count_U = AUC_eval_class_count(Ypred_S, Ypred_U, label_S, label_U, Y)
    Y_correct_S = (Ypred_S == Y).astype(float)
    Y_correct_U = (Ypred_U == Y).astype(float)
    bias = max_S - max_U
    loc_B = np.argsort(bias)
    _, unique_bias_loc = np.unique(sorted(bias), return_index=True)
    unique_bias_loc = unique_bias_loc[1:]
    unique_bias_loc = np.append(unique_bias_loc, num_inst) - 1    
    bias = np.array(sorted(set(bias)))
    # efficient evaluation
    acc_change_S = np.divide(Y_correct_S[loc_B], class_count_S[loc_S[loc_B]] + 1e-12) / L_S
    acc_change_U = np.divide(Y_correct_U[loc_B], class_count_U[loc_U[loc_B]] + 1e-12) / L_U
    AUC_record[:, 0] = np.concatenate([np.array([0]), np.cumsum(-acc_change_S)]) + np.mean(class_correct_S / (class_count_S  + 1e-12))
    AUC_record[:, 1] = np.concatenate([np.array([0]), np.cumsum(acc_change_U)])
    AUC_record = AUC_record[np.concatenate([np.array([0]), unique_bias_loc.reshape(-1)+ 1]), :]
    AUC_record[AUC_record < 0] = 0
    # Compute AUC
    AUC_val = np.trapz(AUC_record[:, 0], AUC_record[:, 1])    
    return AUC_val, AUC_record

def Compute_biasedHM(score_S, score_U, Y, label_S, label_U, fixed_bias):
    # fixed_bias is a list input
    Y = Y.reshape(-1)
    label_S = label_S.reshape(-1)
    label_U = label_U.reshape(-1)    
    AUC_record = np.zeros((Y.shape[0] + 1, 2))
    label_S = np.unique(label_S)
    label_U = np.unique(label_U)
    L_S = label_S.shape[0]
    L_U = label_U.shape[0]
    num_inst = score_S.shape[0]

    # effective bias searching
    loc_S = np.argmax(score_S, axis=1)
    max_S = score_S[np.arange(num_inst), loc_S]
    Ypred_S = label_S[loc_S]
    loc_U = np.argmax(score_U, axis=1)
    max_U = score_U[np.arange(num_inst), loc_U]    
    Ypred_U = label_U[loc_U]
    class_correct_S, class_correct_U, class_count_S, class_count_U = AUC_eval_class_count(Ypred_S, Ypred_U, label_S, label_U, Y)
    Y_correct_S = (Ypred_S == Y).astype(float)
    Y_correct_U = (Ypred_U == Y).astype(float)
    bias = max_S - max_U
    loc_B = np.argsort(bias)
    _, unique_bias_loc = np.unique(sorted(bias), return_index=True)
    unique_bias_loc = unique_bias_loc[1:]
    unique_bias_loc = np.append(unique_bias_loc, num_inst) - 1    
    bias = np.array(sorted(set(bias)))
    # efficient evaluation
    acc_change_S = np.divide(Y_correct_S[loc_B], class_count_S[loc_S[loc_B]] + 1e-12) / L_S
    acc_change_U = np.divide(Y_correct_U[loc_B], class_count_U[loc_U[loc_B]] + 1e-12) / L_U
    AUC_record[:, 0] = np.concatenate([np.array([0]), np.cumsum(-acc_change_S)]) + np.mean(class_correct_S / (class_count_S + 1e-12))
    AUC_record[:, 1] = np.concatenate([np.array([0]), np.cumsum(acc_change_U)])
    AUC_record = AUC_record[np.concatenate([np.array([0]), unique_bias_loc.reshape(-1)+ 1]), :]
    AUC_record[AUC_record < 0] = 0
    acc_noBias = AUC_record[sum(bias <= 0), :]
    # Compute Harmonic mean
    HM_nobias = Compute_HM(acc_noBias)    
    accs = [AUC_record[sum(bias <= f_bias), :] for f_bias in fixed_bias]
    HM = [Compute_HM(acc) for acc in accs]
    return HM, HM_nobias, accs, acc_noBias


## ------------------------GFSL Training Arguments Related ------------------------ 
def postprocess_args(args):
    save_path1 = args.test_mode + '-' + '-'.join([args.dataset, args.model_class, args.backbone_class, '{:02d}w{:02d}s{:02}q'.format(args.way, args.shot, args.query)])
    save_path2 = '_'.join([str('_'.join(args.step_size.split(','))), str(args.gamma),
                           'lr{:.2g}'.format(args.lr),
                           'btz{}'.format(args.batch_size),                           
                           str(args.lr_scheduler), str(args.temperature), 
                           'ntask{}nclass{}T{}'.format(args.num_tasks, args.sample_class, args.temperature),
                           #'gpu{}'.format(args.gpu) if args.multi_gpu else 'gpu0',
                           # str(time.strftime('%Y%m%d_%H%M%S'))
                           ])    
    if args.init_weights is not None:
        save_path1 += '-Pre'
    if args.augment:
        save_path2 += '-Aug'
    if args.lr_mul > 1.0:
        save_path2 += 'lrmul{:.2g}'.format(args.lr_mul)
    if args.model_class in ['Castle', 'ACastle']:
        save_path2 += 'dp{}-h{}'.format(args.dp_rate, args.head)   

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)    
    if not os.path.exists(os.path.join(args.save_dir, save_path1)):
        os.mkdir(os.path.join(args.save_dir, save_path1))
    args.save_path = os.path.join(args.save_dir, save_path1, save_path2)
    return args

def get_command_line_parser():
    parser = argparse.ArgumentParser()
    # task configurations
    parser.add_argument('--sample_class', type=int, default=8)       
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--eval_way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--eval_shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--eval_query', type=int, default=15)
    parser.add_argument('--num_tasks',   type=int, default=3)
    parser.add_argument('--test_mode', type=str, default='FSL', choices=['FSL', 'GFSL'])   # important
    
    # optimization parameters
    parser.add_argument('--max_epoch', type=int, default=200)    
    parser.add_argument('--batch_size', type=int, default=32)    
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_mul', type=float, default=1)    
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['multistep', 'step', 'cosine'])
    parser.add_argument('--step_size', type=str, default='2,4,6,8')
    parser.add_argument('--gamma', type=float, default=0.5)    
    parser.add_argument('--augment',   action='store_true', default=False)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--init_weights', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=1)
    
    # model parameters
    parser.add_argument('--model_class', type=str, default='Castle', choices=['Castle', 'ACastle'])
    parser.add_argument('--backbone_class', type=str, default='Res12', choices=['Res12'])
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['MiniImageNet', 'TieredImageNet'])
    
    # Castle-relate model parameters
    parser.add_argument('--head', type=int, default=1)
    parser.add_argument('--dp_rate', type=float, default=0.1)    
    
    # usually untouched parameters
    parser.add_argument('--orig_imsize', type=int, default=-1)                              # -1 for no cache, and -2 for no resize    
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--eval_interval', type=int, default=200)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--num_eval_episodes', type=int, default=10000)
    
    return parser


## ------------------------GFSL Evaluation Arguments Related ------------------------ 
def postprocess_eval_args(args):
    assert(args.model_path is not None)
    return args

def get_eval_command_line_parser():
    parser = argparse.ArgumentParser()
    # basic configurations
    parser.add_argument('--eval_way', type=int, default=5)
    parser.add_argument('--eval_shot', type=int, default=1)
    parser.add_argument('--eval_query', type=int, default=15)    
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--model_path', type=str, default=None)
    # criteria related
    parser.add_argument('--criteria', type=str, help='A list contains criteria from [Acc, HMeanAcc, HMeanMAP, Delta, AUSUC]', 
                        default='Acc, HMeanAcc, HMeanMAP, Delta, AUSUC') # 
    
    # model parameters
    parser.add_argument('--model_class', type=str, default='Castle', choices=['CLS', 'ProtoNet', 'Castle', 'ACastle'])
    parser.add_argument('--backbone_class', type=str, default='Res12', choices=['Res12'])
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['MiniImageNet', 'TieredImageNet'])
    
    # Castle-relate model parameters
    parser.add_argument('--head', type=int, default=1)
    parser.add_argument('--dp_rate', type=float, default=0.1)    
    
    # usually untouched parameters
    parser.add_argument('--orig_imsize', type=int, default=-1)                              # -1 for no cache, and -2 for no resize    
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_eval_episodes', type=int, default=500)    
    parser.add_argument('--temperature', type=float, default=1)
    return parser