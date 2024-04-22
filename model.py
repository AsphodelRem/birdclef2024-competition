import os
import librosa
import numpy as np
import pandas as pd

import lightning as L
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
import torch.nn as nn
from torch.optim import AdamW, Adam
import torch.nn.functional as F
import torchmetrics

import timm
from config import Config


class ModelUtils:
    @staticmethod
    def compute_class_weights(primary_labels):
        unique_labels = primary_labels.unique()
        total_samples = primary_labels.shape[0]
        class_weights = {}
        for label in unique_labels:
            count = (primary_labels == label).sum() 
            class_weight = total_samples / (len(unique_labels) * count)
            class_weights[label] = class_weight
        return torch.tensor(list(class_weights.values()))

class BirdCLEFModel(L.LightningModule):
    def __init__(self, config: Config):
        super(BirdCLEFModel, self).__init__()
        self.config = config
        self.model = self._create_model()
        self.metadata = pd.read_csv(self.config.metadata)
        self.weights = None # ModelUtils.compute_class_weights(self.metadata['primary_label'])
        self.criterion = nn.CrossEntropyLoss(weight=self.weights)

        self.f1 = torchmetrics.F1Score(task='binary', num_classes=self.config.num_classes, average='macro')
        self.precision = torchmetrics.Precision(task='binary', num_classes=self.config.num_classes, average='macro')
        self.recall = torchmetrics.Recall(task='binary', num_classes=self.config.num_classes, average='macro')

        self.save_hyperparameters()
    
    # TODO: Add loading from a checkpoint
    def update_model(self, new_config: Config, checkpoint: str=None):
        self.config = new_config

    def _create_model(self):
        model = timm.create_model(self.config.model_name, pretrained=True)
        model.classifier = torch.nn.Linear(model.num_features, self.config.num_classes)
        return model

    def forward(self, x):
        return self.model(x)

    def step(self, batch, stage: str):
        x, y = batch
        predict = self.model(x)
        loss = self.criterion(predict, y)

        precision = self.precision(predict, y)
        recall = self.recall(predict, y)
        f1 = self.f1(predict, y)

        self.log(f'{stage}_loss', loss, prog_bar=True)
        self.log(f'{stage}_precision', precision, prog_bar=True)
        self.log(f'{stage}_recall', recall, prog_bar=True)
        self.log(f'{stage}_f1', f1, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')

    def test_step(self, batch, batch_idx):
        self.step(batch, 'test')

    def validation_step(self, batch, batch_idx):
        self.step(batch, 'val')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.config.epochs, T_mult=1, eta_min=1e-6, last_epoch=-1)
        return {
            'optimizer': optimizer, 
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'epoch', 
                'monitor': 'val_loss', 
                'frequency': 1
                }
            }
