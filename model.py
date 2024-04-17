import os
import librosa
import numpy as np
import pandas as pd
import lightning as L

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import torch.nn as nn
from torch.optim import AdamW, Adam
import torch.nn.functional as F
import torchmetrics

import timm

from config import Config

class FocalLoss(L.LightningModule):
    def __init__(self, alpha=1, gamma=2, reduction='mean', weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weights = weights

    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        loss = torch.pow(ce_loss * ((1 - p_t), self.gamma))

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            pass

class BirdCLEFModel(L.LightningModule):
    def __init__(self, config: Config):
        super(BirdCLEFModel, self).__init__()

        self.config = config

        self.model = timm.create_model(config.model_name, pretrained=True)
        self.model.classifier = torch.nn.Linear(self.model.num_features, config.num_classes)

        self.f1 = torchmetrics.F1Score(task='binary', num_classes=config.num_classes, average='macro')
        self.precision = torchmetrics.Precision(task='binary', num_classes=config.num_classes, average='macro')
        self.recall = torchmetrics.Recall(task='binary', num_classes=config.num_classes, average='macro')
        self.aug_roc = torchmetrics.AUROC(task='binary', num_classes=config.num_classes, average='macro')

        self.metadata = pd.read_csv(self.config.metadata)

        weights = self._compute_class_weights(self.metadata['label'])
        self.criterion = nn.CrossEntropyLoss(weight=weights)

        self.save_hyperparameters()
        
    def forward(self, x):
        return self.model(x)
    
    def step(self, batch, stage: str):
        x, y = batch
        predict = self.model(x)
        loss = self.criterion(predict, y)
        
        auc_roc = self.aug_roc(predict, y)
        precision = self.precision(predict, y)
        recall = self.recall(predict, y)
        f1 = self.f1(predict, y)

        self.log(f'{stage}_loss', loss, prog_bar=True)
        self.log(f'{stage}_auc_roc', auc_roc, prog_bar=True)
        self.log(f'{stage}_precision', precision, prog_bar=True)
        self.log(f'{stage}_recall', recall, prog_bar=True)
        self.log(f'{stage}_f1', f1, prog_bar=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')

    def test_step(self, batch, batch_idx):
        self.step(batch, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)

        lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.config.epochs,
            T_mult=1,
            eta_min=1e-6,
            last_epoch=-1
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'epoch',
                'monitor': 'val_loss',
                'frequency': 1
            }
        }
    

    def _compute_class_weights(self, series):
      unique_labels = series.unique()
      total_samples = series.shape[0]
      class_weights = {}
      for label in unique_labels:
          count = (series == label).sum() 
          class_weight = total_samples / (len(unique_labels) * count)
          class_weights[label] = class_weight

      class_weight = sorted(class_weights.items(), key=lambda x: x[0])
      return torch.tensor(list(class_weights.values()))
    