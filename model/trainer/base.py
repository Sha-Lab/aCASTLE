import abc
import torch
import os.path as osp

from model.utils import (
    ensure_path,
    Averager, Timer, count_acc,
    compute_confidence_interval,
)
from model.logger import Logger

class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        ensure_path(
            self.args.save_path,
            scripts_to_save=['model/models', 'model/networks', __file__],
        )
        self.logger = Logger(args, osp.join(args.save_path))

        self.train_step = 0
        self.train_epoch = 0
        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()

        # train statistics
        self.trlog = {}
        self.trlog['max_acc'] = 0.0
        self.trlog['max_acc_epoch'] = 0
        self.trlog['max_acc_interval'] = 0.0

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def evaluate_fsl(self, data_loader):
        pass
    
    @abc.abstractmethod
    def evaluate_gfsl(self, data_loader):
        pass    
    
    @abc.abstractmethod
    def evaluate_test(self):
        pass  
        
    @abc.abstractmethod
    def final_record(self):
        pass    

    def try_evaluate(self, epoch):
        # for validation (evaluation during training)
        args = self.args
        if (self.train_step) % args.eval_interval == 0:
            if args.test_mode == 'GFSL':
                # GFSL test
                vl, va, vap = self.evaluate_gfsl(self.val_fsl_loader, self.val_gfsl_loader, self.trainvalset)
            else:
                # FSL test
                vl, va, vap = self.evaluate_fsl(self.val_fsl_loader)
            self.logger.add_scalar('val_loss', float(vl), self.train_step)
            self.logger.add_scalar('val_acc', float(va),  self.train_step)
            print('epoch {}-{}, val, loss={:.4f} acc={:.4f}+{:.4f}'.format(epoch, self.train_step, vl, va, vap))

            if va >= self.trlog['max_acc']:
                self.trlog['max_acc'] = va
                self.trlog['max_acc_interval'] = vap
                self.trlog['max_acc_epoch'] = self.train_epoch
                self.save_model('max_acc')

    def try_logging(self, tl1, tl2, ta, tg=None):
        args = self.args
        if (self.train_step - 1) % args.log_interval == 0:
            print('epoch {}, train {:06g}/{:06g}, total loss={:.4f}, loss={:.4f} acc={:.4f}, lr={:.4g}'
                  .format(self.train_epoch,
                          self.train_step,
                          self.max_steps,
                          tl1.item(), tl2.item(), ta.item(),
                          self.optimizer.param_groups[0]['lr']))
            self.logger.add_scalar('train_total_loss', tl1.item(), self.train_step)
            self.logger.add_scalar('train_loss', tl2.item(), self.train_step)
            self.logger.add_scalar('train_acc',  ta.item(), self.train_step)
            if tg is not None:
                self.logger.add_scalar('Additional',  tg.item(), self.train_step)
            print('data_timer: {:.2f} sec, '     \
                  'forward_timer: {:.2f} sec,'   \
                  'backward_timer: {:.2f} sec, ' \
                  'optim_timer: {:.2f} sec'.format(
                        self.dt.item(), self.ft.item(),
                        self.bt.item(), self.ot.item())
                  )
            self.logger.dump()

    def save_model(self, name):
        torch.save(
            dict(params=self.model.state_dict()),
            osp.join(self.args.save_path, name + '.pth')
        )

    def __str__(self):
        return "{}({})".format(
            self.__class__.__name__,
            self.model.__class__.__name__
        )
