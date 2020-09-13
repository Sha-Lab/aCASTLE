import abc
import torch
import os.path as osp

from model.utils import (
    ensure_path,
    Averager, Timer, count_acc,
    compute_confidence_interval,
)

class Evaluator(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        # train statistics
        self.trlog = {}
        self.best_bias = None
        self.criteria = [e.lower().strip() for e in args.criteria.split(',')]
    
    @abc.abstractmethod
    def evaluate_gfsl(self, data_loader):
        pass    
        
    @abc.abstractmethod
    def final_record(self):
        pass

    def __str__(self):
        return "{}({})".format(
            self.__class__.__name__,
            self.model.__class__.__name__
        )
