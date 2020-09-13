import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from model.dataloader.samplers import CategoriesSampler, RandomSampler, ClassSampler
from model.models.castle import Castle
from model.models.acastle import ACastle
from model.models.protonet import ProtoNet
from model.models.classifier import Classifier as CLS

def get_dataloader(args):
    if args.dataset == 'MiniImageNet':
        # Handle MiniImageNet
        from model.dataloader.mini_imagenet import MiniImageNet as Dataset
        args.dropblock_size = 5        
    elif args.dataset == 'TieredImageNet':
        from model.dataloader.tiered_imagenet import tieredImageNet as Dataset    
        args.dropblock_size = 5
    else:
        raise ValueError('Non-supported Dataset.')

    num_workers =  args.num_workers
    trainset = Dataset('train', args, augment=False)
    args.num_class = trainset.num_class
    train_gfsl_loader = None  
    train_fsl_loader = None   
    proto_sampler = ClassSampler(trainset.label, 100)
    proto_loader = DataLoader(dataset=trainset, 
                              batch_sampler=proto_sampler, 
                              num_workers=num_workers, 
                              pin_memory=True)     
    
    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label,
                                    500,
                                    min(args.eval_way, valset.num_class), 
                                    args.eval_shot + args.eval_query)
    val_fsl_loader = DataLoader(dataset=valset,
                                batch_sampler=val_sampler,
                                num_workers=num_workers,
                                pin_memory=True)
    
    testset = Dataset('test', args)
    test_sampler = CategoriesSampler(testset.label, 
                                     args.num_eval_episodes,
                                     min(args.eval_way, testset.num_class), args.eval_shot + args.eval_query)
    test_fsl_loader = DataLoader(dataset=testset,
                                 batch_sampler=test_sampler,
                                 num_workers=num_workers,
                                 pin_memory=True)            
    
    # prepare data loaders for GFSL test
    trainvalset = Dataset('aux_val', args) 
    val_many_shot_sampler = RandomSampler(trainvalset.label,
                                          500, 
                                          min(args.eval_way, valset.num_class) * args.eval_query)
    val_gfsl_loader = DataLoader(dataset=trainvalset, 
                                 batch_sampler=val_many_shot_sampler, 
                                 num_workers=num_workers, 
                                 pin_memory=True)    
    
    traintestset = Dataset('aux_test', args) 
    test_many_shot_sampler = RandomSampler(traintestset.label,
                                           args.num_eval_episodes, 
                                           min(args.eval_way, testset.num_class) * args.eval_query)
    test_gfsl_loader = DataLoader(dataset=traintestset, 
                                  batch_sampler=test_many_shot_sampler, 
                                  num_workers=num_workers, 
                                  pin_memory=True)        
    return trainset, valset, trainvalset, testset, traintestset, train_fsl_loader, train_gfsl_loader, proto_loader, val_fsl_loader, val_gfsl_loader, test_fsl_loader, test_gfsl_loader   

def prepare_model(args):
    model = eval(args.model_class)(args)
    pre_dict = torch.load(args.model_path)['params']
    pre_dict = {k:v for k,v in pre_dict.items() if k in model.state_dict()}
    model.load_state_dict(pre_dict)     

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model.cuda()
    if torch.cuda.device_count() > 1:
        model.encoder = nn.DataParallel(model.encoder)
    return model