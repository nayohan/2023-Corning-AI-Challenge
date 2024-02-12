import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as torch_utils
from torch.cuda.amp import autocast

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import losses

from copy import deepcopy
from scipy.stats import spearmanr
from contrastive_learning.loss import *
from contrastive_learning.utils import *

from transformers.models.bert.modeling_bert import BertLMPredictionHead
VERBOSE_SILENT=0
VERBOSE_EPOCH_WISE=1
VERBOSE_BATCH_WISE=2

class CL(Engine):
    def __init__(self, func, model, optimizer, scheduler, args, config, lm_head):
        self.model=model
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.args=args
        self.config=config
        self.criterion=nn.CrossEntropyLoss()
        
        super().__init__(func) 
        
        self.best_corr=0
        self.best_model=None
        self.device = next(model.parameters()).device
        
        self.lm_head = lm_head
        self.scaler = torch.cuda.amp.GradScaler()

    @staticmethod
    def train(engine, mini_batch):
        engine.model.train()
        engine.optimizer.zero_grad()

        input_ids = mini_batch['input_ids'].to(engine.device)
        attention_mask = mini_batch['attention_mask'].to(engine.device)

        input_ids_ = mini_batch['input_ids_'].to(engine.device)
        attention_mask_ = mini_batch['attention_mask_'].to(engine.device)
        
        if engine.args.fp16:
            with autocast(): 
                documents = engine.model(input_ids, attention_mask, output_hidden_states=True, return_dict=None)
                queries = engine.model(input_ids_, attention_mask_, output_hidden_states=True, return_dict=None)
        else:
            documents = engine.model(input_ids, attention_mask, output_hidden_states=True, return_dict=None)
            queries = engine.model(input_ids_, attention_mask_, output_hidden_states=True, return_dict=None)
        
        pooler = Pooler(engine.args.pooler_type)
    
        d = pooler(attention_mask, documents)
        q = pooler(attention_mask_, queries)

        mlp = MLPLayer().to(engine.device)
        z1, z2 = mlp(d), mlp(q)
    
        align = align_loss(z1, z2)
        uniform = uniform_loss(torch.cat([z1, z2]))
        sim = Similarity(engine.args.temp)

        cos_sim = sim(z1.unsqueeze(1), z2.unsqueeze(0))        
        labels = torch.arange(cos_sim.size(0)).long().to(engine.device)
        
        loss = engine.criterion(cos_sim, labels)
            
        engine.scaler.scale(loss).backward()
        engine.scaler.step(engine.optimizer)
        engine.scaler.update()
                    
        return {
            'loss': float(loss),
            'align': float(align),
            'uniform': float(uniform),
        }

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            input_ids = mini_batch['input_ids'].to(engine.device)
            attention_mask = mini_batch['attention_mask'].to(engine.device)

            input_ids_ = mini_batch['input_ids_'].to(engine.device)
            attention_mask_ = mini_batch['attention_mask_'].to(engine.device)
            
            score = mini_batch['score'].long().to(engine.device)
            
            with autocast():
                sent1 = engine.model(input_ids,
                                    attention_mask,
                                    output_hidden_states=True,
                                    return_dict=None,
                                    )
                
                sent2 = engine.model(
                                    input_ids_,
                                    attention_mask_,
                                    output_hidden_states=True,
                                    return_dict=None,
                                    )            
            
            pooler = Pooler('cls_before_pooler')            
            c1 = pooler(attention_mask, sent1)
            c2 = pooler(attention_mask_, sent2)
        
            e_align = align_loss(c1, c2)
            e_uniform = uniform_loss(torch.cat([c1, c2]))
            
            sim = Similarity(temp=1)
            cos_sim = sim(c1.unsqueeze(1), c2.unsqueeze(0))
            stsb_spearman = spearmanr(torch.diag(cos_sim).cpu().numpy(), score.cpu().numpy()).correlation

        return {
            'stsb_spearman': stsb_spearman * 100,
            'e_align': float(e_align),
            'e_uniform': float(e_uniform),
        }
        
    @staticmethod
    def attach(train_engine, validation_engine, verbose=2):

        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine,
                metric_name,
            )
        training_metric_names = ['loss', 'align', 'uniform']

        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)

        if verbose >= verbose:
            pbar = ProgressBar(bar_format=None, ncols=100)
            pbar.attach(train_engine, training_metric_names)

        if verbose >= 1:
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                print('\n<{} epoch>'.format(engine.state.epoch), end='\n')
                print('|Training| loss={:.4f} align={:.3f} uniform={:.3f}'.format(
                    engine.state.metrics['loss'],
                    engine.state.metrics['align'],
                    engine.state.metrics['uniform'],
                ), end='\n')

        validation_metric_names = ['stsb_spearman', 'e_align', 'e_uniform']
        
        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)

        if verbose >= 2:
            pbar = ProgressBar(bar_format=None, ncols=100)
            pbar.attach(validation_engine, validation_metric_names)

        if verbose >= 1:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                print('|Evaluation| stsb_spearman={:.3f} align={:.3f} uniform={:.3f}'.format(
                    engine.state.metrics['stsb_spearman'],
                    engine.state.metrics['e_align'],
                    engine.state.metrics['e_uniform'],
                ), end='\n')
    @staticmethod
    def check_best(engine):
        corr = float(engine.state.metrics['stsb_spearman'])
        if corr >= engine.best_corr: 
            engine.best_corr = corr  
            engine.best_model = deepcopy(engine.model.state_dict())
            print('** Model parameter is updated **', end=' \n')
            torch.save({'model_best': engine.best_model, 'args': engine.args,}, engine.args.output_path)
            print('** A new model is saved to "{}" **'.format(engine.args.output_path))

            
class Trainer():
    def __init__(self, args, config):
        self.args=args
        self.config=config
        
    def train(self,
              model,
              optimizer,
              scheduler,
              train_loader, 
              valid_loader,
    ):
        device = torch.device(self.args.gpu_id)
        self.lm_head = BertLMPredictionHead(self.config).to(device)

        train_engine = CL(
            func=CL.train,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=self.args,
            config=self.config,
            lm_head=self.lm_head,
        )
        valid_engine = CL(
            func=CL.validate,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            args=self.args, 
            config=self.config,
            lm_head=self.lm_head,
        )
        CL.attach(
            train_engine,
            valid_engine,
            verbose=self.args.verbose,
        )
        def run_validation(engine, validation_engine, valid_loader):
            valid_engine.run(valid_loader, max_epochs=1)
        
        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            run_validation, 
            valid_engine, valid_loader,
        )
        valid_engine.add_event_handler(
            Events.EPOCH_COMPLETED, 
            CL.check_best,
        )            
        train_engine.run(
            train_loader,
            max_epochs=self.args.n_epochs,
        )

        return model
