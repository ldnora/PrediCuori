#!/usr/bin/env python
# coding: utf-8

# # Modelo Híbrido DenseNet121 + MLP
#
#
# ## Objetivo do Script_09 no Escopo Silver KNN
#
# O script_09, em sua versão original, itera sobre os 5 datasets treinando um modelo por dataset. Na configuração
# Silver KNN deste roteiro, o loop é simplificado para treinar apenas 1 modelo — o SILVER_KNN — avaliado no test
# set canônico do GOLD. Isso reduz o tempo total de execução em aproximadamente 80%.
# 5.3 Adaptação do CONFIG para Rodar Apenas Silver KNN
#
# ```python
# # Em CONFIG['datasets'] (script_09_modelo_hibrido.py):
# # Manter apenas o SILVER_KNN no dicionário:
# 'datasets': {
# 'GOLD' : {'file': '../csv/ecg_gold_completo_classified.csv',
# 'n': 3013, 'imputacao_pct': 0.0, 'metodo': 'Sem imputacao'},
# 'SILVER_KNN': {'file': '../csv/ecg_silver_knn_imputado_classified.csv',
# 'n': 3481, 'imputacao_pct': 1.67, 'metodo': 'KNN'},
# # Remover ou comentar BRONZE_HYBRID, BRONZE_MICE, BRONZE_KNN
# },
# # ATENÇÃO: GOLD deve permanecer no dicionário pois gerar_analise_comparativa()
# # faz referência a todos_metricas['GOLD'] para calcular Delta_GOLD.
# # Ou adapte a função para calcular delta em relação ao próprio SILVER_KNN.
# ```
#
# > Alternativa simplificada:
# >
# > Se o objetivo é apenas treinar e avaliar o SILVER KNN sem comparação com GOLD, pode-se também remover o GOLD do dicionário e simplificar a função gerar_analise_comparativa() para não calcular Delta_GOLD. Nesse caso, os resultados serão apresentados apenas para o SILVER KNN.
#
#
# ## Arquitetura do Modelo Híbrido
#
# A arquitetura HybridECGClassifier integra dois branches especializados fundidos por concatenação:
#
# | Componente | Input → Output                 | Descrição                                                                                                   |
# | ---------- | ------------------------------ | ----------------------------------------------------------------------------------------------------------- |
# | Branch CNN | `[B, 1, 272, 512] → [B, 1024]` | DenseNet121 pré-treinado em ImageNet, com `conv0` adaptado para 1 canal usando a média dos pesos originais. |
# | Branch MLP | `[B, 14] → [B, 32]`            | Rede densa composta por `Linear(14→64)` + `ReLU` + `Dropout(0.4)` + `Linear(64→32)`.                        |
# | Fusão      | `[B, 1056] → [B, 2]`           | `CONCAT([1024, 32])` seguido de `FC(1056→256→128→2)` + `Dropout` + `Softmax`.                               |
#
#
#
#
# ## Estratégia de Treinamento em 2 Fases
#
# ### Fase 1 — Warm-up (10 épocas, CNN congelada)
#
# - CNN (DenseNet121) completamente congelada — apenas MLP e camadas de fusão treinadas
# - LR único: 1e-3 (otimizador Adam)
# - Objetivo: inicializar os pesos da fusão sem destruir representações ImageNet da CNN
#
#
# ### Fase 2 — Fine-tuning (40 épocas, rede completa)
#
# - Todos os parâmetros liberados para treinamento
# - LRs diferenciados: CNN: 1e-5 | MLP: 1e-4 | Fusão: 1e-4
# - ReduceLROnPlateau: fator 0.5, paciência 5 épocas no val_loss
# - Early stopping: paciência 10 épocas no val_AUC
# - Gradient clipping: grad_clip=1.0 para estabilidade numérica
# - Class weights: calculados sobre y_train do SILVER KNN via compute_class_weight
#
#
# ## Métricas de Avaliação
#
# O modelo é avaliado no test set canônico do GOLD (452 registros). As métricas calculadas são:
# - AUC-ROC: principal métrica — meta >= 0,95
# - Accuracy: acurácia geral no test set
# - F1 Macro: média dos F1 de ambas as classes
# - F1 NORMAL (classe 0) e F1 ANORMAL (classe 1): análise por classe
# - Baseline de referência: XGBoost tabular puro — AUC = 0,928
#
#
# ## Execução
#
# - Garantir que o test set canônico do GOLD está disponível
# ```bash
# ls splits/test_indices.npy # deve existir e conter 452 índices GOLD
# ```
#
# > Tempo estimado: 30–90 minutos (CPU) | 10–25 minutos (GPU NVIDIA com CUDA). Com SILVER KNN e apenas
# > 1 modelo (sem os 4 datasets adicionais), o tempo é reduzido ~80% em relação ao script original.
#
#
# ## Artefatos Gerados
#
# | Artefato                     | Local                    | Conteúdo                                                                   |
# | ---------------------------- | ------------------------ | -------------------------------------------------------------------------- |
# | `modelo_SILVER_KNN_final.pt` | `checkpoints/`           | Pesos finais do modelo, métricas, configurações e scaler.                  |
# | `historico_SILVER_KNN.json`  | `resultados_e_metricas/` | Histórico de treino contendo loss e AUC por época (warm-up + fine-tuning). |
# | `comparative_results.csv`    | `resultados_e_metricas/` | Tabela comparativa com métricas do(s) dataset(s) treinado(s).              |
# | `comparative_roc.png`        | `plots_comparativos/`    | Curva ROC do modelo SILVER KNN comparada ao baseline XGBoost.              |
# | `comparative_auc_barras.png` | `plots_comparativos/`    | Gráfico de barras da AUC-ROC com linha de meta em `0.95`.                  |
# | `script_09_MAIN_*.log`       | `resultados_e_metricas/` | Log completo de execução contendo timestamps e métricas por época.         |
#

# # Config

# In[38]:

from config import build_config_09
from script_08_pytorch_dataset import (
    ECGMultimodalDataset,
    get_train_transform,
    get_eval_transform,
    criar_dataloaders,
    ajustar_scaler,
)
from device_utils import get_device, patch_config_for_device, optimizer_step, save_checkpoint, TPU_AVAILABLE
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    classification_report, roc_curve,
    precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torchvision import models
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn as nn
import torch
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')


# # Configure logging

# In[39]:


def configurar_logging(results_dir, tag='MAIN'):
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(results_dir, f'script_09_{tag}_{ts}.log')
    logger = logging.getLogger(f'rescue2_e7_{tag}')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s  %(levelname)-8s  %(message)s',
                                datefmt='%H:%M:%S')
        fh = logging.FileHandler(log_path, encoding='utf-8')
        ch = logging.StreamHandler(sys.stdout)
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# # Hybrid model -> DenseNet121/ResNet50/EfficientNet_B0 + MLP

# In[40]:


def criar_backbone(backbone_name):
    if backbone_name == 'densenet121':
        # Branch CNN: DenseNet121 pre-treinado com conv0 adaptado 3ch -> 1ch
        densenet = models.densenet121(weights='IMAGENET1K_V1')
        n_cnn = densenet.classifier.in_features  # 1024

        # Inicializacao correta: media dos pesos ImageNet RGB preserva magnitude
        old_w = densenet.features.conv0.weight.data
        new_conv0 = nn.Conv2d(1, 64, kernel_size=7,
                              stride=2, padding=3, bias=False)
        new_conv0.weight.data = old_w.mean(dim=1, keepdim=True)
        densenet.features.conv0 = new_conv0
        densenet.classifier = nn.Identity()
        cnn_branch = densenet

    elif backbone_name == 'resnet50':
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        n_cnn = resnet.fc.in_features  # 2048

        old_w = resnet.conv1.weight.data  # [64, 3, 7, 7]
        new_conv = nn.Conv2d(1, 64, kernel_size=7,
                             stride=2, padding=3, bias=False)
        new_conv.weight.data = old_w.mean(dim=1, keepdim=True)
        resnet.conv1 = new_conv
        resnet.fc = nn.Identity()
        cnn_branch = resnet

    elif backbone_name == 'efficientnet_b0':
        efficientnet = models.efficientnet_b0(weights='IMAGENET1K_V1')
        n_cnn = efficientnet.classifier[1].in_features  # 1280

        old_w = efficientnet.features[0][0].weight.data
        new_conv = nn.Conv2d(1, 32, kernel_size=3,
                             stride=2, padding=1, bias=False)
        new_conv.weight.data = old_w.mean(dim=1, keepdim=True)
        efficientnet.features[0][0] = new_conv
        efficientnet.classifier = nn.Identity()
        cnn_branch = efficientnet

    else:
        raise ValueError(
            f'Backbone {backbone_name} não identificado. Espera-se densenet121, resnet50 ou efficientnet_b0')

    return cnn_branch, n_cnn


# In[41]:


class HybridECGClassifier(nn.Module):
    def __init__(self, config, backbone_name):
        super().__init__()
        dropout = config['dropout']
        n_tab = len(config['param_cols'])
        self.backbone_name = backbone_name
        self.model_type = 'hybrid'
        self.has_pretrained_backbone = True

        self.cnn_branch, n_cnn = criar_backbone(backbone_name)

        # Branch MLP: 14 -> 128 -> 64 -> 32
        self.mlp_branch = nn.Sequential(
            nn.Linear(n_tab, 128), nn.BatchNorm1d(128), nn.ReLU(
                inplace=True), nn.Dropout(dropout),
            nn.Linear(128, 64),   nn.BatchNorm1d(64),  nn.ReLU(
                inplace=True), nn.Dropout(dropout),
            nn.Linear(64, 32),    nn.BatchNorm1d(32),  nn.ReLU(inplace=True),
        )

        # Fusao: [1056] -> [256] -> [128] -> [2]
        self.fusion = nn.Sequential(
            nn.Linear(
                n_cnn + 32, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 128),        nn.BatchNorm1d(
                128), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(128, config['n_classes']),
        )

    def forward(self, image, params):
        cnn_features = self.cnn_branch(image)
        mlp_features = self.mlp_branch(params)
        fused = torch.cat([cnn_features, mlp_features], dim=1)

        return self.fusion(fused)

    def congelar_cnn(self):
        for p in self.cnn_branch.parameters():
            p.requires_grad = False

    def descongelar_cnn(self):
        for p in self.cnn_branch.parameters():
            p.requires_grad = True

    def contar_parametros(self):
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'treinaveis': train, 'congelados': total - train}


# # MLP Model

# In[42]:


class MLPOnlyClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        dropout = config['dropout']
        n_tab = len(config['param_cols'])
        self.model_type = 'mlp'
        self.backbone_name = self.model_type
        self.has_pretrained_backbone = False

        # Branch MLP: 14 -> 128 -> 64 -> 32 -> 2
        self.mlp_branch = nn.Sequential(
            nn.Linear(n_tab, 128), nn.BatchNorm1d(128), nn.ReLU(
                inplace=True), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(
                inplace=True), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(
                inplace=True), nn.Dropout(dropout),
            nn.Linear(32, config['n_classes'])
        )

    def forward(self, image=None, params=None):
        return self.mlp_branch(params)

    def contar_parametros(self):
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'treinaveis': train, 'congelados': total - train}


# # CNN Model -> Densenet-121

# In[43]:


class CNNOnlyClassifier(nn.Module):
    def __init__(self, config, backbone_name='densenet121'):
        super().__init__()
        dropout = config['dropout']
        self.model_type = 'cnn'
        self.backbone_name = backbone_name
        self.has_pretrained_backbone = True

        self.cnn_branch, self.n_cnn = criar_backbone(backbone_name)

        self.classifier = nn.Sequential(
            nn.Linear(self.n_cnn, 256), nn.BatchNorm1d(
                256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, config['n_classes'])
        )

    def forward(self, image, params=None):
        features = self.cnn_branch(image)
        return self.classifier(features)

    def congelar_cnn(self):
        for p in self.cnn_branch.parameters():
            p.requires_grad = False

    def descongelar_cnn(self):
        for p in self.cnn_branch.parameters():
            p.requires_grad = True

    def contar_parametros(self):
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'treinaveis': train, 'congelados': total - train}


# # Epoch

# In[44]:


def executar_epoca(model, loader, criterion, optimizer, device, treino=True, grad_clip=1.0):
    model.train() if treino else model.eval()
    total_loss, labels_l, probs_l, preds_l, n = 0.0, [], [], [], 0

    ctx = torch.enable_grad() if treino else torch.no_grad()
    with ctx:
        for batch in loader:
            imgs = batch['image'].to(
                device, non_blocking=True) if 'image' in batch else None
            params = batch['params'].to(
                device, non_blocking=True) if 'params' in batch else None
            labels = batch['label'].to(device, non_blocking=True)

            if treino:
                optimizer.zero_grad(set_to_none=True)
            logits = model(imgs, params)
            loss = criterion(logits, labels)
            if treino:
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer_step(optimizer)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            total_loss += loss.item()
            labels_l.append(labels.cpu().numpy())
            probs_l.append(probs.detach().cpu().numpy())
            preds_l.append(preds.cpu().numpy())
            n += 1

    y, p, yp = (np.concatenate(x) for x in [labels_l, probs_l, preds_l])
    return {
        'loss': total_loss / max(n, 1),
        'accuracy': float(accuracy_score(y, yp)),
        'auc_roc': float(roc_auc_score(y, p)),
        'f1': float(f1_score(y, yp, average='macro', zero_division=0)),
    }


# # Early stopping

# In[45]:


class EarlyStopping:
    def __init__(self, patience, ckpt_path, logger):
        self.patience, self.ckpt_path, self.logger = patience, ckpt_path, logger
        self.best_auc = -1.0
        self.sem_melhoria = 0

    def __call__(self, val_auc, model):
        if val_auc > self.best_auc + 1e-5:
            self.best_auc = val_auc
            self.sem_melhoria = 0
            save_checkpoint(model.state_dict(), self.ckpt_path)
            self.logger.info(f'    Checkpoint salvo (val AUC={val_auc:.4f})')
        else:
            self.sem_melhoria += 1
            if self.sem_melhoria >= self.patience:
                self.logger.info('    Early stopping.')
                return True
        return False


# # Treimento em 2 fases para os híbridos

# ## Fase 1: Warm-up

# In[46]:


def fase_warmup(model, train_loader, val_loader, criterion, config, device, ckpt_dir, logger):
    logger.info(
        f'\n  FASE 1 — WARM-UP | {config["warmup_epochs"]} epocas | CNN congelada')
    model.congelar_cnn()
    info = model.contar_parametros()
    logger.info(
        f'  Parametros treinaveis: {info["treinaveis"]:,}/{info["total"]:,}')

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters(
    )), lr=config['warmup_lr'], weight_decay=config['weight_decay'])
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', patience=config['lr_patience'], factor=config['lr_factor'])
    es = EarlyStopping(config['early_stop_patience'],
                       os.path.join(ckpt_dir, 'best_warmup.pt'), logger)

    hist = []
    for ep in range(1, config['warmup_epochs'] + 1):
        t0 = time.time()
        tr = executar_epoca(model, train_loader, criterion,
                            opt, device, True, config['grad_clip'])
        vl = executar_epoca(model, val_loader, criterion, opt, device, False)
        dt = time.time() - t0
        sched.step(vl['auc_roc'])
        logger.info(f'  E{ep:02d}/{config["warmup_epochs"]:02d} '
                    f'tr_auc={tr["auc_roc"]:.4f} tr_loss={tr["loss"]:.4f} | '
                    f'val_auc={vl["auc_roc"]:.4f} val_loss={vl["loss"]:.4f} | {dt:.0f}s')
        hist.append({'fase': 'warmup', 'epoca': ep,
                    'train': tr, 'val': vl, 'tempo_s': dt})
        if es(vl['auc_roc'], model):
            break

    model.load_state_dict(torch.load(os.path.join(
        ckpt_dir, 'best_warmup.pt'), weights_only=True))
    logger.info(f'  Melhor val AUC-ROC (fase 1): {es.best_auc:.4f}')

    return hist


# ## Fase 2: Fine tunning

# In[47]:


def fase_finetuning(model, train_loader, val_loader, criterion, config, device, ckpt_dir, logger):
    logger.info(
        f'\n  FASE 2 — FINE-TUNING | ate {config["finetune_epochs"]} epocas | rede completa')
    model.descongelar_cnn()

    if model.model_type == 'hybrid':
        opt = optim.AdamW([
            {'params': model.cnn_branch.parameters(), 'lr': config['lr_cnn']},
            {'params': model.mlp_branch.parameters(), 'lr': config['lr_mlp']},
            {'params': model.fusion.parameters(), 'lr': config['lr_fusion']},
        ], weight_decay=config['weight_decay'])
    elif model.model_type == 'cnn':
        opt = optim.AdamW([
            {'params': model.cnn_branch.parameters(), 'lr': config['lr_cnn']},
        ], weight_decay=config['weight_decay'])
    else:
        opt = optim.AdamW([
            {'params': model.mlp_branch.parameters(), 'lr': config['lr_mlp']},
        ], weight_decay=config['weight_decay'])

    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', patience=config['lr_patience'], factor=config['lr_factor'])
    es = EarlyStopping(config['early_stop_patience'], os.path.join(
        ckpt_dir, 'best_finetune.pt'), logger)

    hist = []
    for ep in range(1, config['finetune_epochs'] + 1):
        t0 = time.time()
        tr = executar_epoca(model, train_loader, criterion,
                            opt, device, True,  config['grad_clip'])
        vl = executar_epoca(model, val_loader,   criterion, opt, device, False)
        dt = time.time() - t0
        sched.step(vl['auc_roc'])
        logger.info(f'  E{ep:02d}/{config["finetune_epochs"]:02d} '
                    f'tr_auc={tr["auc_roc"]:.4f} tr_loss={tr["loss"]:.4f} | '
                    f'val_auc={vl["auc_roc"]:.4f} val_loss={vl["loss"]:.4f} | {dt:.0f}s')
        hist.append({'fase': 'finetune', 'epoca': ep,
                    'train': tr, 'val': vl, 'tempo_s': dt})
        if es(vl['auc_roc'], model):
            break

    model.load_state_dict(torch.load(
        os.path.join(ckpt_dir, 'best_finetune.pt'), weights_only=True))
    logger.info(f'  Melhor val AUC-ROC (fase 2): {es.best_auc:.4f}')

    return hist


# # Treinamento para a MLP

# In[48]:

def fase_treinamento_mlp(model, train_loader, val_loader, criterion, config, device, ckpt_dir, logger):
    logger.info(
        f'\n  FASE TREINAMENTO MLP | ate {config["finetune_epochs"]} epocas | rede completa')

    opt = optim.AdamW([
        {'params': model.mlp_branch.parameters(), 'lr': config['lr_mlp']},
    ], weight_decay=config['weight_decay'])

    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', patience=config['lr_patience'], factor=config['lr_factor'])
    es = EarlyStopping(config['early_stop_patience'], os.path.join(
        ckpt_dir, 'best_finetune.pt'), logger)

    hist = []

    for ep in range(1, config['finetune_epochs'] + 1):
        t0 = time.time()
        tr = executar_epoca(model, train_loader, criterion,
                            opt, device, True,  config['grad_clip'])
        vl = executar_epoca(model, val_loader, criterion, opt, device, False)
        dt = time.time() - t0
        sched.step(vl['auc_roc'])
        logger.info(f'  E{ep:02d}/{config["finetune_epochs"]:02d} '
                    f'tr_auc={tr["auc_roc"]:.4f} tr_loss={tr["loss"]:.4f} | '
                    f'val_auc={vl["auc_roc"]:.4f} val_loss={vl["loss"]:.4f} | {dt:.0f}s')
        hist.append({'fase': 'treinamento', 'epoca': ep,
                    'train': tr, 'val': vl, 'tempo_s': dt})
        if es(vl['auc_roc'], model):
            break

    model.load_state_dict(torch.load(
        os.path.join(ckpt_dir, 'best_finetune.pt'), weights_only=True))
    logger.info(f'  Melhor val AUC-ROC (fase 2): {es.best_auc:.4f}')

    return hist


# # Avaliar test set

# In[49]:


def avaliar_test_set(model, model_name, test_loader, device, dataset_name, logger):
    logger.info(f'\n  AVALIACAO FINAL — test set canonico ({dataset_name})')
    model.eval()
    labels_l, probs_l, preds_l = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch['image'].to(
                device, non_blocking=True) if 'image' in batch else None
            params = batch['params'].to(
                device, non_blocking=True) if 'params' in batch else None
            labels = batch['label'].to(device, non_blocking=True)
            logits = model(imgs, params)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)
            labels_l.append(labels.cpu().numpy())
            probs_l.append(probs.cpu().numpy())
            preds_l.append(preds.cpu().numpy())

    y_true = np.concatenate(labels_l)
    y_prob = np.concatenate(probs_l)
    y_pred = np.concatenate(preds_l)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    m = {
        'model_name': model_name,
        'n_test': int(len(y_true)),
        'auc_roc': float(roc_auc_score(y_true, y_prob)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
        'f1_normal': float(f1_score(y_true, y_pred, pos_label=0, average='binary')),
        'f1_anormal': float(f1_score(y_true, y_pred, pos_label=1, average='binary')),
        'precision_normal': precision_score(y_true, y_pred, pos_label=0, average='binary'),
        'precision_anormal': precision_score(y_true, y_pred, pos_label=1, average='binary'),
        'recall_normal': recall_score(y_true, y_pred, pos_label=0, average='binary'),
        'recall_anormal': recall_score(y_true, y_pred, pos_label=1, average='binary'),
        'roc_fpr': fpr.tolist(),
        'roc_tpr': tpr.tolist(),
        'y_true': y_true.tolist(),
        'y_prob': y_prob.tolist(),
    }

    logger.info(f'  AUC-ROC  : {m["auc_roc"]:.4f}')
    logger.info(f'  Accuracy : {m["accuracy"]:.4f}')
    logger.info(f'  F1 macro : {m["f1_macro"]:.4f}')
    logger.info(classification_report(y_true, y_pred,
                target_names=['NORMAL', 'ANORMAL'], digits=4))
    return m


# # Criar modelo

# In[50]:


def criar_modelo(model_name, config):
    if model_name == 'densenet121_hybrid':
        return HybridECGClassifier(config, backbone_name='densenet121')
    elif model_name == 'resnet50_hybrid':
        return HybridECGClassifier(config, backbone_name='resnet50')
    elif model_name == 'efficientnet_b0_hybrid':
        return HybridECGClassifier(config, backbone_name='efficientnet_b0')
    elif model_name == 'cnn_only':
        return CNNOnlyClassifier(config)
    elif model_name == 'mlp_only':
        return MLPOnlyClassifier(config)
    else:
        raise ValueError(
            f'O {model_name} não é um modelo aceito. Aceita apenas densenet121, resnet50, efficientnet_b0, cnn ou mlp')


# In[51]:


def carregar_splits(splits_dir: str) -> dict:
    train_path = os.path.join(splits_dir, 'train_path.npy')
    val_path = os.path.join(splits_dir, 'val_path.npy')
    gold_test_path = os.path.join(splits_dir, 'gold_test_path.npy')

    for split_path in [train_path, val_path, gold_test_path]:
        if not os.path.exists(split_path):
            raise FileNotFoundError(
                f'Split nao encontrado: {split_path}\n'
                'Execute o script_08 antes de prosseguir.')

    train_idx = np.load(train_path)
    val_idx = np.load(val_path)
    gold_test_idx = np.load(gold_test_path)

    return train_idx, val_idx, gold_test_idx


# In[52]:

def teste_sanidade(model, train_loader, device, config, logger):
    logger.info('\n  TESTES DE SANIDADE')
    batch = next(iter(train_loader))
    erros = []

    if model.model_type in ['cnn', 'hybrid']:
        imgs = batch['image'].to(device)
        exp = (config['batch_size'], config['img_channels']) + \
            tuple(config['img_resize'])
        if tuple(imgs.shape) != exp:
            erros.append(f'Shape imagem: {tuple(imgs.shape)} != {exp}')
        else:
            logger.info(f'  [1] Shape imagem  : {tuple(imgs.shape)}  OK')
    else:
        imgs = None

    if model.model_type in ['mlp', 'hybrid']:
        params = batch['params'].to(device)
        if params.shape[1] != len(config['param_cols']):
            erros.append(f'Shape params: {params.shape}')
        else:
            logger.info(f'  [2] Shape params  : {tuple(params.shape)}  OK')
    else:
        params = None

    model.eval()
    with torch.no_grad():
        logits = model(imgs, params)

    if logits.shape != (config['batch_size'], config['n_classes']):
        erros.append(f'Shape logits: {logits.shape}')
    else:
        logger.info(f'  [3] Shape logits  : {tuple(logits.shape)}  OK')

    if torch.isnan(logits).any() or torch.isinf(logits).any():
        erros.append('Logits contem nan/inf')
    else:
        logger.info(f'  [4] Logits validos: OK')

    logger.info(
        f'  [5] Parametros    : {model.contar_parametros()["total"]:,}  OK')

    if erros:
        for e in erros:
            logger.error(f'  ERRO: {e}')
        raise RuntimeError('Testes de sanidade falharam.')
    logger.info('  Todos os testes passaram.\n')


def treinar_modelo(model, model_name, train_loader, val_loader, gold_test_loader, criterion, config, device, logger):
    ckpt_dir = os.path.join(config['checkpoints_dir'], model_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    if model.model_type == 'mlp':
        hist = fase_treinamento_mlp(
            model, train_loader, val_loader, criterion, config, device, ckpt_dir, logger)
    else:
        hist_w = fase_warmup(model, train_loader, val_loader,
                             criterion, config, device, ckpt_dir, logger)
        hist_ft = fase_finetuning(
            model, train_loader, val_loader, criterion, config, device, ckpt_dir, logger)
        hist = hist_w + hist_ft

    with open(os.path.join(config['results_dir'], f'historico_{model_name}.json'),
              'w', encoding='utf-8') as f:
        json.dump(hist, f, indent=2)

    metricas = avaliar_test_set(
        model, model_name, gold_test_loader, device, model_name, logger)

    # Serializar modelo final
    save_checkpoint({
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'config': {k: v for k, v in config.items() if k != 'datasets'},
        'metricas_test': {k: v for k, v in metricas.items()
                          if k not in ('roc_fpr', 'roc_tpr', 'y_true', 'y_prob')},
    }, os.path.join(ckpt_dir, f'modelo_{model_name}_final.pt'))

    return metricas


def gerar_analise_comparativa(todos_metricas, config, logger):
    plots_dir = os.path.join(config['results_dir'], 'plots_comparativos')
    os.makedirs(plots_dir, exist_ok=True)

    rows = []

    for model_name, m in todos_metricas.items():
        rows.append({
            'Modelo': model_name,
            'AUC-ROC': round(m['auc_roc'], 4),
            'Accuracy': round(m['accuracy'], 4),
            'F1 Macro': round(m['f1_macro'], 4),
            'F1 Normal': round(m['f1_normal'], 4),
            'F1 Anormal': round(m['f1_anormal'], 4),
            'Precisao Normal': round(m['precision_normal'], 4),
            'Precisao Anormal': round(m['precision_anormal'], 4),
            'Recall Normal': round(m['recall_normal'], 4),
            'Recall Anormal': round(m['recall_anormal'], 4),
        })

    df_comp = (
        pd.DataFrame(rows)
        .sort_values('AUC-ROC', ascending=False)
        .reset_index(drop=True)
    )

    df_comp.to_csv(
        os.path.join(config['results_dir'], 'comparative_results.csv'),
        index=False,
        encoding='utf-8'
    )

    with open(
        os.path.join(config['results_dir'], 'comparative_results.json'),
        'w',
        encoding='utf-8'
    ) as f:

        json.dump(
            {
                model_name: {
                    k: v for k, v in metricas.items()
                    if k not in ('roc_fpr', 'roc_tpr', 'y_true', 'y_prob')
                }
                for model_name, metricas in todos_metricas.items()
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    cores = {
        'densenet121_hybrid': '#1F3864',
        'resnet50_hybrid': '#2E75B6',
        'efficientnet_b0_hybrid': '#70AD47',
        'cnn_only': '#ED7D31',
        'mlp_only': '#FFC000',
    }

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(
        [0, 1],
        [0, 1],
        'k--',
        lw=1,
        alpha=0.4,
        label='Aleatório (AUC=0.500)'
    )

    for model_name, m in todos_metricas.items():

        cor = cores.get(model_name, None)

        ax.plot(
            np.array(m['roc_fpr']),
            np.array(m['roc_tpr']),
            lw=2,
            color=cor,
            label=f'{model_name} (AUC={m["auc_roc"]:.4f})'
        )

    ax.set_xlabel('Taxa de Falsos Positivos')
    ax.set_ylabel('Taxa de Verdadeiros Positivos')

    ax.set_title(
        'Curvas ROC - Comparação dos Modelos\n'
        'Avaliação no GOLD TEST'
    )

    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    fig.tight_layout()

    fig.savefig(
        os.path.join(plots_dir, 'comparative_roc.png'),
        dpi=150
    )

    plt.close(fig)

    modelos = list(df_comp['Modelo'])
    aucs = [todos_metricas[m]['auc_roc'] for m in modelos]

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(
        range(len(modelos)),
        aucs,
        color=[cores.get(m, '#808080') for m in modelos],
        alpha=0.85,
        edgecolor='white'
    )

    for bar, auc in zip(bars, aucs):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            auc + 0.002,
            f'{auc:.4f}',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )

    ax.set_xticks(range(len(modelos)))
    ax.set_xticklabels(modelos, rotation=20)

    ax.set_ylabel('AUC-ROC')

    ax.set_title(
        'Comparação de AUC-ROC entre Modelos\n'
        'Avaliação no GOLD TEST'
    )

    ax.grid(axis='y', alpha=0.3)

    ymin = max(0.0, min(aucs) - 0.03)
    ax.set_ylim([ymin, 1.0])

    fig.tight_layout()

    fig.savefig(
        os.path.join(plots_dir, 'comparative_auc_barras.png'),
        dpi=150
    )

    plt.close(fig)

    melhor_modelo = max(
        todos_metricas,
        key=lambda m: todos_metricas[m]['auc_roc']
    )

    pior_modelo = min(
        todos_metricas,
        key=lambda m: todos_metricas[m]['auc_roc']
    )

    delta_auc = (
        todos_metricas[melhor_modelo]['auc_roc'] -
        todos_metricas[pior_modelo]['auc_roc']
    )

    logger.info('\n=== COMPARAÇÃO FINAL ===')
    logger.info(
        f'Melhor modelo : {melhor_modelo} '
        f'(AUC={todos_metricas[melhor_modelo]["auc_roc"]:.4f})'
    )
    logger.info(
        f'Pior modelo   : {pior_modelo} '
        f'(AUC={todos_metricas[pior_modelo]["auc_roc"]:.4f})'
    )
    logger.info(
        f'Delta AUC     : {delta_auc:.4f} '
        f'({delta_auc*100:.2f}%)'
    )

    logger.info(
        f'Limiar H0     : < {config["delta_threshold"] * 100:.0f}%'
    )

    if delta_auc < config['delta_threshold']:
        logger.info(
            'CONCLUSÃO: H0 CONFIRMADA — as arquiteturas apresentam '
            'desempenho equivalente no conjunto GOLD TEST.'
        )
    else:
        logger.info(
            'CONCLUSÃO: H0 REJEITADA — existe diferença relevante '
            'de desempenho entre as arquiteturas avaliadas.'
        )

    logger.info(
        f'\nResultado final:\n{df_comp.to_string(index=False)}'
    )

    return df_comp


# In[53]:

def main():
    config = build_config_09()
    device = get_device()
    config = patch_config_for_device(config, device)

    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    os.makedirs(config['results_dir'], exist_ok=True)
    os.makedirs(config['checkpoints_dir'], exist_ok=True)

    logger = configurar_logging(config['results_dir'], 'MAIN')
    logger.info(
        f'\n{"="*70}\n'
        f'  SCRIPT 09\n'
        f'  Estudo Comparativo: 1 Datasets x 5 Arquiteturas\n'
        f'  {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\n'
        f'{"="*70}'
    )

    logger.info(f'  Device: {device}')

    todos_metricas = {}
    t_inicio = time.time()

    train_idx, val_idx, gold_test_idx = carregar_splits(config['splits_dir'])

    scaler = ajustar_scaler(
        train_idx, config['datasets']['SILVER']['file'], config['param_cols'])

    y_train = pd.read_csv(config['datasets']['SILVER']['file'])[
        config['label_col']].values[train_idx]
    weights = compute_class_weight(
        'balanced', classes=np.array([0, 1]), y=y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(
        weights, dtype=torch.float32).to(device))

    for i, (model_name, mode) in enumerate(config['models_modes'].items(), 1):
        logger.info(f'\n{"="*70}')
        logger.info(f'  INICIANDO O {i}º MODELO: {model_name}')
        logger.info(f'{"="*70}')

        t_ds = time.time()

        train_loader, val_loader, gold_test_loader = criar_dataloaders(
            train_idx, val_idx, gold_test_idx, scaler, config, mode
        )

        model = criar_modelo(model_name=model_name, config=config)

        model.to(device)

        teste_sanidade(model, train_loader, device, config, logger)

        metricas = treinar_modelo(model, model_name, train_loader,
                                  val_loader, gold_test_loader, criterion, config, device, logger)

        todos_metricas[model_name] = metricas

        logger.info(f'  {model_name} concluido em {(time.time()-t_ds)/60:.1f} min '
                    f'| AUC={metricas["auc_roc"]:.4f}')
        logger.info(f'\n{"="*70}')

    # Analise comparativa
    df_comp = gerar_analise_comparativa(todos_metricas, config, logger)

    t_total = (time.time() - t_inicio) / 60
    logger.info(f'\n{"="*70}')
    logger.info(f'  SCRIPT 09 CONCLUIDO em {t_total:.1f} min')
    logger.info(f'{"="*70}')


if __name__ == '__main__':
    main()
