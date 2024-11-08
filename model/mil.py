import copy, random
import itertools
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc, RocCurveDisplay

from omegaconf import OmegaConf
from hydra.utils import instantiate

import torch
import torch.nn.functional as F
import wandb

from monai.metrics import ROCAUCMetric, ConfusionMatrixMetric, get_confusion_matrix, compute_confusion_matrix_metric
from monai.networks.utils import one_hot

from mislight.networks.utils import load_pretrained_net
from mislight.models import BaseModel
from mislight.models.utils import instantiate_scheduler
from mislight.utils.misc import label2colormap

from utils.misc import get_current_rampup_weight, update_ema_variables, tuple_index

class MILModel_Base(BaseModel):
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
            self.net_names = ['netB', 'netA', 'netC']
        else:
            self.net_names = ['netB', 'netA', 'netC']
                    
        for net in self.net_names:
            setattr(self, net, instantiate(OmegaConf.select(opt['networks'], net), _convert_='partial'))
            pretrained = OmegaConf.select(opt['networks'], f'pretrained.{net}')
            if pretrained:
                snet = getattr(self, net)
                snet = load_pretrained_net(snet, pretrained)
                
    def define_losses(self, **opt):
        if self.is_train:            
            self.criterionCls = instantiate(opt['loss_cls'], _convert_='partial')        
                   
    def define_etc(self, **opt):   
        pass
                 
    def set_input(self, batch):
        self.image = batch['image']
        if 'clabel' in batch.keys():
            self.clabel = self.pp_clabel = batch['clabel'].long()
            # self.labeled = batch['label'].ge(0).ravel()
            
            #ty = batch['label']
            #ty[:,0] += 1e-7
            #ty = one_hot(ty.argmax(1, keepdim=True), ty.shape[1])
            #self.label = ty
            
    def convert_output(self, x):
        return torch.softmax(x, 1)
        #return one_hot(x.argmax(1, keepdim=True), x.shape[1])
    
    def _step_forward(self, batch, batch_idx):
        self.set_input(batch)
        
        self.image_cls = self._forward_train(self.image)
        
    def _step_forward_infer(self, batch, batch_idx):
        self.set_input(batch)
        
        #outputs = self.inferer(self.image, self._forward_train)
        outputs = self.forward(self.image)
        if hasattr(self, 'clabel'):
            self.pp_clabel = F.one_hot(self.clabel, outputs.shape[1])
        return outputs
      
    def _forward_train(self, x):
        mb = self.hparams['minibatch']
        x = x.permute(0,4,1,2,3)
        logits = []
        for i in range(int(np.ceil(x.shape[1] / float(mb)))):
            x_slice = x[:,i*mb : (i+1)*mb]
            sh = x_slice.shape
            x_slice = x_slice.reshape(sh[0]*sh[1], *sh[2:])
            logits_slice = self.netB(x_slice)
            logits_slice = logits_slice.reshape(sh[0], sh[1], -1)
            logits.append(logits_slice)
        logits = torch.cat(logits, dim=1)
        agg = self.netA(logits)
        out = self.netC(agg)
        return out
    
    ### pl methods
    def forward(self, x):
        out = self._forward_train(x)
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
        loss = self.criterionCls(outputs, self.clabel)

        if hasattr(self, 'metrics'):
            pp_outputs = self.convert_output(outputs)
            for k in self.metrics.keys():
                self.metrics[k](pp_outputs.float(), self.pp_clabel.float())
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
        self.pp_outputs = self.convert_output(outputs)
        return self.pp_outputs
    
    def test_step(self, batch, batch_idx):
        outputs = self._step_forward_infer(batch, batch_idx)
        self.outputs = outputs
        
        if hasattr(self, 'metrics'):
            pp_outputs = self.convert_output(outputs)
            for k in self.metrics.keys():
                self.metrics[k](pp_outputs.float(), self.pp_clabel.float())
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

class MILModel(MILModel_Base): 
    def training_step(self, batch, batch_idx):
        stage = 'train'
        self._step_forward(batch, batch_idx)
        
        loss = 0
        
        # Labeled
        bs = self.image.size(0)                               
        # Cls loss: S(A) ~ Ya
        w0 = self.hparams['lambda_cls']
        if w0 > 0:            
            loss_S = self.criterionCls(self.image_cls, self.clabel)
            self.log('loss/cls', loss_S, batch_size=bs, on_step=True, on_epoch=True)
        else:
            loss_S = 0
        loss += loss_S * w0  

        # 
        #self.log(f'loss/{stage}', loss, batch_size=bs, on_step=True, on_epoch=True)

        return loss
