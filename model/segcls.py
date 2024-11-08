import copy, random
import itertools
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc, RocCurveDisplay

from omegaconf import OmegaConf
from hydra.utils import instantiate

import torch
from torch.distributions.uniform import Uniform
import wandb

from monai.metrics import ROCAUCMetric, ConfusionMatrixMetric, get_confusion_matrix, compute_confusion_matrix_metric
from monai.networks.utils import one_hot

from mislight.networks.utils import load_pretrained_net
from mislight.models import BaseModel
from mislight.models.utils import instantiate_scheduler
from mislight.utils.misc import label2colormap

from utils.misc import get_current_rampup_weight, update_ema_variables, tuple_index

class ImageSegClsModel_Base(BaseModel):
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
            self.net_names = ['netS','netC']
        else:
            self.net_names = ['netS','netC']
                    
        for net in self.net_names:
            setattr(self, net, instantiate(OmegaConf.select(opt['networks'], net), _convert_='partial'))
            pretrained = OmegaConf.select(opt['networks'], f'pretrained.{net}')
            if pretrained:
                snet = getattr(self, net)
                snet = load_pretrained_net(snet, pretrained)
                
    def define_losses(self, **opt):
        if self.is_train:            
            self.criterionSeg = instantiate(opt['loss_seg'], _convert_='partial')
            self.criterionCls = instantiate(opt['loss_cls'], _convert_='partial')        
                   
    def define_etc(self, **opt):   
        pass
                 
    def set_input(self, batch):
        self.imageA = batch['imageA']
        if 'labelA' in batch.keys():
            ty = batch['labelA']
            ty[:,0] += 1e-7
            ty = one_hot(ty.argmax(1, keepdim=True), ty.shape[1])
            self.labelA = self.pp_labelA = ty
        if 'clabelA' in batch.keys():
            self.clabelA = self.pp_clabelA = batch['clabelA'].long()
            self.clabeledA = batch['clabelA'].ge(0).ravel()
        if 'imageB' in batch.keys():
            self.imageB = batch['imageB']
        if 'clabelB' in batch.keys():
            self.clabelB = self.pp_clabelB = batch['clabelB'].long()
            self.clabeledB = batch['clabelB'].ge(0).ravel()
            
    def convert_output(self, x):
        return one_hot(x.argmax(1, keepdim=True), x.shape[1])
    
    def convert_coutput(self, x):
        return torch.softmax(x, 1)
    
    def _step_forward(self, batch, batch_idx):
        self.set_input(batch)
        
        self.imageA_seg, self.imageA_cls = self._forward_train(self.imageA, return_aux=True)
        self.imageB_seg, self.imageB_cls = self._forward_train(self.imageB, return_aux=True)
        
    def _step_forward_infer(self, batch, batch_idx):
        self.set_input(batch)
        
        outputs = self.inferer(self.imageA, self._forward_train)
        #outputs = self.forward(self.image)
        return outputs
          
    def _forward_train(self, x, return_aux=False):
        if return_aux:
            out, out_feat = self.netS(x, return_aux=return_aux)
            out_feat = self.netC(out_feat)
            return out, out_feat
        else:
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
        
        bs = self.imageA.size(0)  
        loss = self.criterionSeg(outputs, self.labelA)

        if hasattr(self, 'metrics'):
            pp_outputs = self.convert_output(outputs)
            for k in self.metrics.keys():
                self.metrics[k](pp_outputs.float(), self.pp_labelA.float())
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
                self.metrics[k](pp_outputs.float(), self.pp_labelA.float())
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

class ImageSegClsModel_FSL(ImageSegClsModel_Base): 
    def training_step(self, batch, batch_idx):
        stage = 'train'
        self._step_forward(batch, batch_idx)
        
        loss = 0
        
        # Labeled
        bs = self.imageA.size(0)                               
        # Seg loss: S(A) ~ Ya
        w0 = self.hparams['lambda_seg']
        if w0 > 0:            
            loss_S = self.criterionSeg(self.imageA_seg, self.labelA)
            self.log('loss/seg', loss_S, batch_size=bs, on_step=True, on_epoch=True)
        else:
            loss_S = 0
        loss += loss_S * w0  

        # 
        #self.log(f'loss/{stage}', loss, batch_size=bs, on_step=True, on_epoch=True)

        return loss
    
    
class ImageSegClsModel_GC(ImageSegClsModel_Base):
    def _get_current_rampup(self):
        rampup = 1
        w0 = self.hparams['lambda_aux']
        if w0 > 0:
            current_time = self.current_epoch if ('rampup_Tmax_epoch' in self.hparams and self.hparams['rampup_Tmax_epoch']) else self.global_step
            max_time = self.hparams['rampup_Tmax'] if 'rampup_Tmax' in self.hparams else 0
            rampup = get_current_rampup_weight(current_time, max_time)
            self.log('Charts/rampup', rampup, on_step=True, on_epoch=False)
        self._current_rampup = rampup       
       
    def define_losses(self, **opt):
        if self.is_train:            
            self.criterionSeg = instantiate(opt['loss_seg'], _convert_='partial')
            self.criterionCls = instantiate(opt['loss_cls'], _convert_='partial')
            self.criterionAux = instantiate(opt['loss_aux'], _convert_='partial')
        
    def training_step(self, batch, batch_idx):
        stage = 'train'
        self._step_forward(batch, batch_idx)
        
        self._get_current_rampup()
        
        loss = 0
        
        # Labeled
        bs = self.imageA.size(0)                               
        # Seg loss: S(A) ~ Ya
        w0 = self.hparams['lambda_seg']
        if w0 > 0:            
            loss_S = self.criterionSeg(self.imageA_seg, self.labelA)
            self.log('loss/seg', loss_S, batch_size=bs, on_step=True, on_epoch=True)
        else:
            loss_S = 0
        loss += loss_S * w0  

        # Labeled - Aux loss
        w0 = self.hparams['lambda_aux']
        w = w0 * self._current_rampup 
        if w > 0:
            #rota = random.choice([(2,3),(3,4),(2,4)])
            rota = (2,3)
            rotn = random.choice([1,2,3])
            
            x_rot = torch.rot90(self.imageA, k=rotn, dims=rota)
            pseg_rot, pcls_rot = self._forward_train(x_rot, return_aux=True)           
            y_rot = torch.rot90(self.imageA_seg, k=rotn, dims=rota)
            
            loss_a = self.criterionAux(pseg_rot, y_rot)
            
            self.log('loss/Aaux', loss_a, batch_size=bs, on_step=True, on_epoch=True)
        else:
            loss_a = 0
        loss += loss_a * w 
        
        # UnLabeled - Aux loss
        w0 = self.hparams['lambda_aux']
        w = w0 * self._current_rampup 
        if w > 0:
            #rota = random.choice([(2,3),(3,4),(2,4)])
            rota = (2,3)
            rotn = random.choice([1,2,3])
            
            x_rot = torch.rot90(self.imageB, k=rotn, dims=rota)
            pseg_rot, pcls_rot = self._forward_train(x_rot, return_aux=True)           
            y_rot = torch.rot90(self.imageB_seg, k=rotn, dims=rota)
            
            loss_a = self.criterionAux(pseg_rot, y_rot)
            
            self.log('loss/Baux', loss_a, batch_size=bs, on_step=True, on_epoch=True)
        else:
            loss_a = 0
        loss += loss_a * w 

        # 
        #self.log(f'loss/{stage}', loss, batch_size=bs, on_step=True, on_epoch=True)

        return loss



### Enc Dec

class ImageSegClsModelEncDec_Base(ImageSegClsModel_Base):
    def define_networks(self, **opt):
        if self.is_train:
            self.net_names = ['netE','netD','netC']
        else:
            self.net_names = ['netE','netD','netC']
                    
        for net in self.net_names:
            setattr(self, net, instantiate(OmegaConf.select(opt['networks'], net), _convert_='partial'))
            pretrained = OmegaConf.select(opt['networks'], f'pretrained.{net}')
            if pretrained:
                snet = getattr(self, net)
                snet = load_pretrained_net(snet, pretrained)

    def _forward_train(self, x, returns=False):
        skips = self.netE(x)
        out = self.netD(skips)
        if returns:
            auxfeat = skips[-1]
            auxout = self.netC(auxfeat)
            return out, auxout
        return out
        
    def _step_forward(self, batch, batch_idx):
        self.set_input(batch)
        
        self.imageA_seg, self.imageA_cls = self._forward_train(self.imageA, returns=True)
        self.imageB_seg, self.imageB_cls = self._forward_train(self.imageB, returns=True)
        
    def _step_forward_infer(self, batch, batch_idx):
        self.set_input(batch)
        
        outputs = self.inferer(self.imageA, self._forward_train)
        #outputs = self.forward(self.image)
        return outputs
    
class ImageSegClsModelEncDec_FSL(ImageSegClsModelEncDec_Base): 
    def training_step(self, batch, batch_idx):
        stage = 'train'
        self._step_forward(batch, batch_idx)
        
        loss = 0
        
        # Labeled
        bs = self.imageA.size(0)                               
        # Seg loss: S(A) ~ Ya
        w0 = self.hparams['lambda_seg']
        if w0 > 0:            
            loss_S = self.criterionSeg(self.imageA_seg, self.labelA)
            self.log('loss/seg', loss_S, batch_size=bs, on_step=True, on_epoch=True)
        else:
            loss_S = 0
        loss += loss_S * w0  

        # 
        #self.log(f'loss/{stage}', loss, batch_size=bs, on_step=True, on_epoch=True)

        return loss
    
class ImageSegClsModelEncDec_FSL_noB(ImageSegClsModelEncDec_FSL):
    def _step_forward(self, batch, batch_idx):
        self.set_input(batch)
        
        self.imageA_seg, self.imageA_cls = self._forward_train(self.imageA, returns=True)
        #self.imageB_seg, self.imageB_cls = self._forward_train(self.imageB, returns=True)
    
class ImageSegClsModelEncDec_CCTFeatureNoise_V0(ImageSegClsModelEncDec_Base): 
    def _get_current_rampup(self):
        rampup = 1
        w0 = self.hparams['lambda_aux']
        if w0 > 0:
            current_time = self.current_epoch if ('rampup_Tmax_epoch' in self.hparams and self.hparams['rampup_Tmax_epoch']) else self.global_step
            max_time = self.hparams['rampup_Tmax'] if 'rampup_Tmax' in self.hparams else 0
            rampup = get_current_rampup_weight(current_time, max_time)
            self.log('Charts/rampup', rampup, on_step=True, on_epoch=False)
        self._current_rampup = rampup
        
    def define_etc(self, **opt):        
        uni_range = opt['noise_range'] if 'noise_range' in opt else 0.1
        self._uni_dist = Uniform(-uni_range, uni_range)
        
    def define_losses(self, **opt):
        if self.is_train:            
            self.criterionSeg = instantiate(opt['loss_seg'], _convert_='partial')
            self.criterionCls = instantiate(opt['loss_cls'], _convert_='partial')
            self.criterionAux = instantiate(opt['loss_aux'], _convert_='partial')
        
    def _perturb_feature(self, x):
        noise_vector = self._uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise
        
    def _forward_train(self, x, returns=False):
        skips = self.netE(x)
        out = self.netD(skips)
        
        if returns:
            aux_skips = [self._perturb_feature(z) for z in skips]
            aux_out = self.netD(aux_skips)
            
            cfeat = skips[-1]
            cout = self.netC(cfeat)
            aux_cfeat = aux_skips[-1]
            aux_cout = self.netC(aux_cfeat)
            return out, cout, aux_out, aux_cout
        return out
        
    def _step_forward(self, batch, batch_idx):
        self.set_input(batch)
        
        self.imageA_seg, self.imageA_cls, self.imageA_auxseg, self.imageA_auxcls = self._forward_train(self.imageA, returns=True)
        self.imageB_seg, self.imageB_cls, self.imageB_auxseg, self.imageB_auxcls = self._forward_train(self.imageB, returns=True)
        
    def training_step(self, batch, batch_idx):
        stage = 'train'
        self._step_forward(batch, batch_idx)
        
        self._get_current_rampup()
        
        loss = 0
        
        # Labeled
        bs = self.imageA.size(0)
        # Seg loss: S(A) ~ Ya
        w0 = self.hparams['lambda_seg']
        if w0 > 0:            
            loss_S = self.criterionSeg(self.imageA_seg, self.labelA)
            self.log('loss/seg', loss_S, batch_size=bs, on_step=True, on_epoch=True)
        else:
            loss_S = 0
        loss += loss_S * w0 
        
        # Unlabeled - aux
        bs = self.imageB.size(0)
        w0 = self.hparams['lambda_aux']
        w = w0 * self._current_rampup
        if w > 0:
            loss_a = self.criterionAux(self.imageB_auxseg, self.imageB_seg.detach())
            self.log('loss/Baux', loss_a, batch_size=bs, on_step=True, on_epoch=True)
        else:
            loss_a = 0
        loss += loss_a * w  
        
        return loss
    
    
    
class ImageSegClsModelEncDec_AddCls_FSL(ImageSegClsModelEncDec_Base): 
    def training_step(self, batch, batch_idx):
        stage = 'train'
        self._step_forward(batch, batch_idx)
        
        loss = 0
        
        # Labeled
        bs = self.imageA.size(0)                               
        # Seg loss: S(A) ~ Ya
        w0 = self.hparams['lambda_seg']
        if w0 > 0:            
            loss_S = self.criterionSeg(self.imageA_seg, self.labelA)
            self.log('loss/seg', loss_S, batch_size=bs, on_step=True, on_epoch=True)
        else:
            loss_S = 0
        loss += loss_S * w0  

        # Cls
        bs = self.clabeledA.sum() + self.clabeledB.sum()
        if bs > 0:
            w0 = self.hparams['lambda_cls']
            if w0 > 0:
                image_cls = torch.cat([
                    self.imageA_cls[self.clabeledA],
                    self.imageB_cls[self.clabeledB],
                ], 0)
                clabel = torch.cat([
                    self.clabelA[self.clabeledA],
                    self.clabelB[self.clabeledB],
                ], 0)
                loss_C = self.criterionCls(image_cls, clabel)
                
                pass
            else:
                loss_C = 0
            loss += loss_C * w0
            self.log('loss/cls', loss_S, batch_size=bs, on_step=False, on_epoch=True)

        # 
        #self.log(f'loss/{stage}', loss, batch_size=bs, on_step=True, on_epoch=True)

        return loss