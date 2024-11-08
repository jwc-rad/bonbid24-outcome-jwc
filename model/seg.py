import copy, random
import itertools
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc, RocCurveDisplay

from omegaconf import OmegaConf
from hydra.utils import instantiate

import torch
import wandb

from monai.metrics import ROCAUCMetric, ConfusionMatrixMetric, get_confusion_matrix, compute_confusion_matrix_metric
from monai.networks.utils import one_hot

from mislight.networks.utils import load_pretrained_net
from mislight.models import BaseModel
from mislight.models.utils import instantiate_scheduler
from mislight.utils.misc import label2colormap

from utils.misc import get_current_rampup_weight, update_ema_variables, tuple_index

class ImageSegModel_Base(BaseModel):
    def __init__(self, **opt):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.is_train = opt['train']
        self.use_wandb = 'wandb' in opt['logger']
        
        # define networks
        self.define_networks(**opt)
        
        # define loss functions
        self.define_losses(**opt)

        # define inferer
        if 'inferer' in opt:
            self.inferer = instantiate(opt['inferer'], _convert_='partial')
            
        # define metrics
        if 'metrics' in opt and opt['metrics'] is not None:
            self.metrics = {}
            for k, v in opt['metrics'].items():
                if '_target_' in v:
                    self.metrics[k] = instantiate(v, _convert_='partial')
                    
        self.define_etc(**opt)
                    
    ### custom methods
    
    def define_networks(self, **opt):
        if self.is_train:
            self.net_names = ['netS']
        else:
            self.net_names = ['netS']
                    
        for net in self.net_names:
            setattr(self, net, instantiate(OmegaConf.select(opt['networks'], net), _convert_='partial'))
            pretrained = OmegaConf.select(opt['networks'], f'pretrained.{net}')
            if pretrained:
                snet = getattr(self, net)
                snet = load_pretrained_net(snet, pretrained)
                
    def define_losses(self, **opt):
        if self.is_train:            
            self.criterionSeg = instantiate(opt['loss_seg'], _convert_='partial')        
                   
    def define_etc(self, **opt):   
        pass
                 
    def set_input(self, batch):
        self.image = batch['image']
        if 'label' in batch.keys():
            ty = batch['label']
            ty[:,0] += 1e-7
            ty = one_hot(ty.argmax(1, keepdim=True), ty.shape[1])
            self.label = ty
            
    def convert_output(self, x):
        return one_hot(x.argmax(1, keepdim=True), x.shape[1])
    
    def _step_forward(self, batch, batch_idx):
        self.set_input(batch)
        
        self.image_seg = self._forward_train(self.image)
        
    def _step_forward_infer(self, batch, batch_idx):
        self.set_input(batch)
        
        outputs = self.inferer(self.image, self._forward_train)
        #outputs = self.forward(self.image)
        return outputs
      
    def _forward_train(self, x):
        out = self.netS(x)
        return out
      
    def tta_v1(self, x):
        y = self.inferer(x, self._forward_train)
        cnt = 1
        
        tx = torch.rot90(x, 1, [2,3])
        ty = self.inferer(tx, self._forward_train)
        y += torch.rot90(ty, 3, [2,3])
        cnt += 1
        
        tx = torch.rot90(x, 1, [2,4])
        ty = self.inferer(tx, self._forward_train)
        y += torch.rot90(ty, 3, [2,4])
        cnt += 1
        
        tx = torch.rot90(x, 1, [3,4])
        ty = self.inferer(tx, self._forward_train)
        y += torch.rot90(ty, 3, [3,4])
        cnt += 1
        
        y = y / cnt
        return y
    
    def tta_v2(self, x):
        y = self.inferer(x, self._forward_train)
        cnt = 1
        
        for rotdim in [[2,3], [2,4], [3,4]]:
            for i in range(1,4):
                tx = torch.rot90(x, i, rotdim)
                ty = self.inferer(tx, self._forward_train)
                y += torch.rot90(ty, 4-i, rotdim)
                cnt += 1
        
        y = y / cnt
        return y
    
    ### pl methods
    def forward(self, x):
        out = self.netS(x)
        #out = self.inferer(x, self._forward_train)
        return out
    
    def configure_optimizers(self):        
        netparams = [getattr(self, n).parameters() for n in self.net_names if hasattr(self, n)]
        optimizer_GF = instantiate(self.hparams['optimizer'], params=itertools.chain(*netparams))
        
        optimizers = [optimizer_GF]
        schedulers = [{
            k: instantiate_scheduler(optimizer, v) if k=='scheduler' else v 
            for k,v in self.hparams['scheduler'].items()
        } for optimizer in optimizers]
        
        return optimizers, schedulers
    
    def validation_step(self, batch, batch_idx):
        stage = 'valid'
        outputs = self._step_forward_infer(batch, batch_idx)
        
        bs = self.image.size(0)  
        loss = self.criterionSeg(outputs, self.label)

        if hasattr(self, 'metrics'):
            pp_outputs = self.convert_output(outputs)
            for k in self.metrics.keys():
                self.metrics[k](pp_outputs.float(), self.label.float())
                if self.global_step == 0 and self.use_wandb:
                    for x in k.split('__'):
                        wandb.define_metric(f'metrics/valid_{x}', summary='max')
                    
        self.log(f'loss/{stage}', loss, batch_size=bs, on_step=True, on_epoch=True)
                    
        return loss
    
    def on_validation_epoch_end(self):
        if not hasattr(self, 'metrics'):
            return
    
        for k in self.metrics.keys():
            if self.metrics[k].get_buffer() is not None:
                mean_metric = self.metrics[k].aggregate()
                if isinstance(mean_metric, list):
                    kks = k.split('__')
                    for i in range(len(mean_metric)):
                        mmetric = mean_metric[i].item()
                        self.log(f'metrics/valid_{kks[i]}', mmetric)                    
                else:
                    mean_metric = mean_metric.item()
                    self.log(f'metrics/valid_{k}', mean_metric)
            self.metrics[k].reset()
            
    def predict_step(self, batch, batch_idx):
        outputs = self._step_forward_infer(batch, batch_idx)
        self.outputs = outputs
        return None
    
    def test_step(self, batch, batch_idx):
        outputs = self._step_forward_infer(batch, batch_idx)
        self.outputs = outputs
        
        if hasattr(self, 'metrics'):
            pp_outputs = self.convert_output(outputs)
            for k in self.metrics.keys():
                self.metrics[k](pp_outputs.float(), self.label.float())
        return None
    
    def on_test_epoch_end(self):
        if not hasattr(self, 'metrics'):
            return
    
        for k in self.metrics.keys():
            if self.metrics[k].get_buffer() is not None:
                mean_metric = self.metrics[k].aggregate()
                if isinstance(mean_metric, list):
                    kks = k.split('__')
                    for i in range(len(mean_metric)):
                        mmetric = mean_metric[i].item()
                        self.log(f'test_metrics/{kks[i]}', mmetric)                    
                else:
                    mean_metric = mean_metric.item()
                    self.log(f'test_metrics/{k}', mean_metric)
            self.metrics[k].reset()    

class ImageSegModel(ImageSegModel_Base): 
    def training_step(self, batch, batch_idx):
        stage = 'train'
        self._step_forward(batch, batch_idx)
        
        loss = 0
        
        # Labeled
        bs = self.image.size(0)                               
        # Seg loss: S(A) ~ Ya
        w0 = self.hparams['lambda_seg']
        if w0 > 0:            
            loss_S = self.criterionSeg(self.image_seg, self.label)
            self.log('loss/seg', loss_S, batch_size=bs, on_step=True, on_epoch=True)
        else:
            loss_S = 0
        loss += loss_S * w0  

        # 
        #self.log(f'loss/{stage}', loss, batch_size=bs, on_step=True, on_epoch=True)

        return loss
