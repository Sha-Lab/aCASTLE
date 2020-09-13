import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from model.dataloader.samplers import CategoriesSampler, RandomSampler
from model.models.castlem import CastleM
from model.models.castle import Castle
from model.models.acastle import ACastle

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
    trainset = Dataset('train', args, augment=args.augment)
    args.num_class = trainset.num_class
    train_gfsl_loader = DataLoader(dataset=trainset, 
                                   batch_size=args.batch_size, 
                                   shuffle=True, 
                                   num_workers=num_workers, 
                                   pin_memory=True)        
    train_sampler = CategoriesSampler(trainset.label, 
                                      len(train_gfsl_loader), 
                                      args.sample_class, 
                                      args.shot)
    train_fsl_loader = DataLoader(dataset=trainset, 
                                   batch_sampler=train_sampler, 
                                   num_workers=num_workers, 
                                   pin_memory=True)    
        
    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label,
                                    args.num_eval_episodes,
                                    min(args.eval_way, valset.num_class), 
                                    args.eval_shot + args.eval_query)
    val_fsl_loader = DataLoader(dataset=valset,
                                batch_sampler=val_sampler,
                                num_workers=num_workers,
                                pin_memory=True)
    
    testset = Dataset('test', args)
    test_sampler = CategoriesSampler(testset.label, 
                                     10000, # args.num_eval_episodes,
                                     min(args.eval_way, testset.num_class), args.eval_shot + args.eval_query)
    test_fsl_loader = DataLoader(dataset=testset,
                                 batch_sampler=test_sampler,
                                 num_workers=num_workers,
                                 pin_memory=True)            
    
    if args.test_mode == 'GFSL':
        # prepare data loaders for GFSL test
        trainvalset = Dataset('aux_val', args) 
        val_many_shot_sampler = RandomSampler(trainvalset.label,
                                              args.num_eval_episodes, 
                                              min(args.eval_way, valset.num_class) * args.eval_query)
        val_gfsl_loader = DataLoader(dataset=trainvalset, 
                                     batch_sampler=val_many_shot_sampler, 
                                     num_workers=num_workers, 
                                     pin_memory=True)    
        
        traintestset = Dataset('aux_test', args) 
        test_many_shot_sampler = RandomSampler(traintestset.label,
                                               10000, 
                                               min(args.eval_way, testset.num_class) * args.eval_query)
        test_gfsl_loader = DataLoader(dataset=traintestset, 
                                      batch_sampler=test_many_shot_sampler, 
                                      num_workers=num_workers, 
                                      pin_memory=True)        
        return trainset, valset, trainvalset, testset, traintestset, train_fsl_loader, train_gfsl_loader, val_fsl_loader, val_gfsl_loader, test_fsl_loader, test_gfsl_loader   
    else:        
        return trainset, valset, None, testset, None, train_fsl_loader, train_gfsl_loader, val_fsl_loader, None, test_fsl_loader, None

def prepare_model(args):
    model = eval(args.model_class)(args)

    # load pre-trained model (no FC weights)
    if args.init_weights is not None:
        model_dict = model.state_dict()        
        pretrained_dict = torch.load(args.init_weights)['params']
        pretrained_dict['cls.weight'] = pretrained_dict['fc.weight']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model.cuda()
    if torch.cuda.device_count() > 1:
        model.encoder = nn.DataParallel(model.encoder)
    return model

def prepare_optimizer(model, args, len_train_loader):
    if args.model_class in ['Castle', 'ACastle']:
        top_para = [v for k,v in model.named_parameters() if ('encoder' not in k and 'cls' not in k)]        
        optimizer = optim.SGD(
                        [{'params': model.encoder.parameters()},
                         {'params': top_para, 'lr': args.lr * args.lr_mul}],
                        lr=args.lr,
                        momentum=args.mom,
                        nesterov=True,
                        weight_decay=args.weight_decay
                    )        
    else:
        optimizer = optim.SGD(
                        model.parameters(),
                        lr=args.lr,
                        momentum=args.mom,
                        nesterov=True,
                        weight_decay=args.weight_decay
                    )
    if args.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(
                            optimizer,
                            step_size=int(args.step_size),
                            gamma=args.gamma
                        )
    elif args.lr_scheduler == 'multistep':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
                            optimizer,
                            milestones=[int(_) * len_train_loader for _ in args.step_size.split(',')],
                            gamma=args.gamma,
                        )
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            args.max_epoch  * len_train_loader,
                            eta_min=0   # a tuning parameter
                        )
    else:
        raise ValueError('No Such Scheduler')

    return optimizer, lr_scheduler
