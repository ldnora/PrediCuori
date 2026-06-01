#!/usr/bin/env python
# coding: utf-8

# # Dataset PyTorch e DataLoaders Multimodais
#
#
# ## Objetivo e Função no Pipeline
#
# O script_08_pytorch_dataset.ipynb implementa o ECGMultimodalDataset, a classe central que integra imagens e
# parâmetros tabulares em um único objeto PyTorch Dataset. Além da classe Dataset, o script define os pipelines
# de transforms (augmentação para treino, normalização para validação/teste), realiza os splits estratificados e valida
# os DataLoaders com testes de sanidade.
#
#
# ## Adaptação para Silver KNN
# O script_08, em sua versão original, usa o dataset GOLD como base dos splits. Para usar o SILVER KNN, a
# única alteração necessária é no campo dataset_file do CONFIG:
#
# ```python
# # Alterar em CONFIG (script_08_pytorch_dataset.ipynb):
# # VERSÃO ORIGINAL (GOLD — NÃO usar neste roteiro):
# # 'dataset_file': '../csv/ecg_gold_completo_classified.csv',
# # VERSÃO SILVER KNN (usar neste roteiro):
# 'dataset_file': '../csv/ecg_silver_knn_imputado_classified.csv',
# ```
#
# <br>
#
#
# >   Por que o SILVER KNN como base dos splits?
# >
# >   Ao usar o SILVER KNN (3.481 registros) como base, o split de treino conterá mais exemplos do que o GOLD (3.013), aproveitando os registros com imputação KNN de baixa intensidade (1,67%). O testset canônico (452 registros do GOLD) é carregado pelo script_09 independentemente — o script_08 não precisa saber do test set do GOLD.
#
#
#
# ## Splits Estratificados com Silver KNN
#
#
# | Split | Proporção | N aprox. | Uso |
# | :--- | :---: | :---: | :--- |
# | **Train** | 70% | ~2.437 registros | Treino do modelo + fit do `StandardScaler` |
# | **Validation** | 15% | ~522 registros | Monitoramento de early stopping e ajuste de LR |
# | **Test (interno)** | 15% | ~522 registros | Reserva — **NOT** usado no `script_09` desta configuração |
# | **Test canônico (GOLD)** | — | 452 registros | Avaliação final no `script_09` (carregado de `test_indices.npy`) |
#
#
# ## Pipeline de Transforms
# O pipeline define dois conjuntos de transforms separados para treino e avaliação:
#
# ### Transforms de Treino (com augmentação conservadora)
#
# - Grayscale(1): garante modo L mesmo em exceções de formato
# - Resize((272, 512)): redimensiona preservando aspect ratio real 3385:1793
# - RandomAffine: translate=0.02, scale=(0.98, 1.02) — shift ±2% e zoom ±2%
# - ColorJitter: brightness=0.1, contrast=0.1 — jitter suave de iluminação
# - ToTensor(): converte para tensor [0, 1]
# - Normalize(mean=[0.5], std=[0.5]): mapeia para faixa [-1, 1]
#
# >    Restrições de augmentação em ECG:
# >    Flip horizontal/vertical e rotações aleatórias são PROIBIDOS em ECG. O eixo X representa tempo (morfologia temporal das ondas P, QRS, T) e o eixo Y representa amplitude (derivações fixas). Qualquer transformação geométrica não-sutil invalida o significado clínico da imagem.
#
#
# ### Transforms de Avaliação (val/test — sem augmentação)
#
# - Grayscale(1) + Resize((272, 512)) + ToTensor() + Normalize(0.5, 0.5)
# - Sem augmentação — avaliação determinística e reproduzível
#
#
#
# ## Normalização Tabular (StandardScaler)
#
# O StandardScaler é ajustado EXCLUSIVAMENTE nos dados de treino (fit apenas no train split) e aplicado por
# transformação (transform) nos splits de validação e teste. Isso previne data leakage da distribuição de validação
# para o fit do scaler.
# As 14 features tabulares normalizadas são: HR, Pd, PR, QRS_Dur, QT, QTC, P_axis, QRS_axis, T_axis, RV5,
# SV1, RV5_SV1_sum, RV6, SV2.
#
#
# ## Execução
#
# Execute célula por célula ou todas de uma única vez.
#
#
# ## Testes de Sanidade
#
# O script executa automaticamente 5 testes de sanidade antes de finalizar:
#
# 1. Tamanho do dataset: N total == N registros no CSV filtrado para SILVER_KNN
# 2. Shape dos tensores de imagem: (1, 272, 512) — canal, altura, largura
# 3. Range dos pixels pós-normalização: valores próximos ao intervalo [-1, 1]
# 4. Throughput dos DataLoaders: mede ms/batch em train, val e test loaders
# 5. Distribuição de classes: verifica balanceamento NORMAL vs. ANORMAL por split
#
#
#
# ## Artefatos Gerados
# | Artefato | Local | Conteúdo |
# | :--- | :--- | :--- |
# | `train_indices.npy` | `splits/` | Índices de treino do SILVER KNN (array numpy) |
# | `val_indices.npy` | `splits/` | Índices de validação do SILVER KNN |
# | `test_indices.npy` | `splits/` | Índices de test interno do SILVER KNN (**NÃO** é o test canônico GOLD) |
# | `dataloader_sanity_report.txt` | `resultados_e_metricas/` | Relatório de sanidade: splits, shapes, throughput |
# | `dataset_sample_grid.png` | `plots_comparativos/` | Grade $4 \times 4$ de amostras com label NORMAL/ANORMAL |
#
#
# <br>
#
# >    Atenção — test_indices.npy gerado pelo script_08 vs. test canônico GOLD:
# >    O script_08 gera um test_indices.npy baseado no SILVER KNN. O script_09, porém, carrega o test
# >    set CANÔNICO do GOLD (452 registros) para avaliação final. Portanto, o test_indices.npy do SILVER
# >    KNN gerado aqui NÃO é usado diretamente na avaliação do script_09. O script_09 exige que o test
# >    set canônico do GOLD (gerado anteriormente em uma execução do script_08 com GOLD) esteja
# >    disponível em splits/test_indices.npy. Veja a primeira seção do script 09
#

# In[1]:


import matplotlib.pyplot as plt
from device_utils import get_device, patch_config_for_device, optimizer_step, save_checkpoint
import os
import re
import time
import warnings
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore')


# # Configurações

# In[2]:


CONFIG = {
    # Paths
    # Dataset SILVER: 3.481 registros, 1,67% imputação com KNN
    'datasets': {
        'GOLD': {'file': 'csv/ecg_gold_completo_classified.csv', 'n': 3013, 'imputacao_pct': 0.0,   'metodo': 'Sem imputacao'},
        'SILVER': {'file': 'csv/ecg_silver_knn_imputado_classified.csv', 'n': 3481, 'imputacao_pct': 1.67,  'metodo': 'KNN'},
    },
    'image_dir': 'image_tracings',
    'splits_dir': 'src/splits/',
    'results_dir': 'resultados_e_metricas/script_09_modelo_hibrido',
    'output_dir':   'resultados_e_metricas/script_08_pytorch_dataset',
    'plots_dir':    'resultados_e_metricas/script_08_pytorch_dataset/plots_comparativos',

    # Parâmetros tabulares (14 features)
    'param_cols': [
        'HR', 'Pd', 'PR', 'QRS_Dur', 'QT', 'QTC',
        'P_axis', 'QRS_axis', 'T_axis',
        'RV5', 'SV1', 'RV5_SV1_sum', 'RV6', 'SV2'
    ],
    'label_col':    'classificacao',
    'filename_col': 'filename',

    # Dimensões das imagens
    # Corpus real: 3385x1793 px (aspect ratio ~1.888:1), modo Gray, 300 DPI.
    # Resize target: 512 largura x 272 altura preserva a proporção original.
    # (512 / 1.888 = 271.2 -> arredondado para 272, múltiplo de 8 para CNNs)
    'img_resize': (136, 256),  # ORIGINAL(272, 512),   # (altura, largura)
    'img_channels': 1,            # grayscale — imagens já chegam em modo L;
                                  # transforms.Grayscale() atua como no-op
                                  # defensivo para garantir consistência

    'train_ratio': 0.70,
    'val_ratio':   0.15,
    # representa 15% do dataset gold dentro do silver
    'test_ratio':  0.173249253235977,
    'random_seed': 42,

    # DataLoader
    'batch_size':  32,
    'num_workers': 4,      # macOS: manter 0 para evitar problemas de fork
                           # Linux/GPU: ajustar para 4-8 conforme hardware
    'pin_memory':  False,  # True apenas com CUDA disponível

    # Data augmentation (treino)
    # Augmentation conservador: ECG tem orientação temporal (eixo x = tempo)
    # e espacial (derivações padronizadas) fixas — flip e rotação são proibidos.
    'aug_translate': 0.02,   # shift máximo de ±2% em x e y
    'aug_scale_min': 0.98,   # zoom mínimo (98%)
    'aug_scale_max': 1.02,   # zoom máximo (102%)
    'aug_brightness': 0.1,   # jitter de brilho ±10%
    'aug_contrast':   0.1,   # jitter de contraste ±10%

    # Normalização: [0,255] -> [-1, 1] via Normalize(mean=0.5, std=0.5)
    'normalize_mean': [0.5],
    'normalize_std':  [0.5],

    # Metadados
    'author':         'Leandro Dalla Nora',
    'institution':    'UFSM - Departamento de Computação Aplicada - Curso de Sistemas de Informação',
    'project':        'PrediCuori'
}


# # Funções utilitárias

# In[3]:


def normalizar_nome_arquivo(filename: str) -> str:
    """
    Normaliza filename para padrão zero-padded .png.
    Extrai o número do identificador e formata com 4 dígitos.
    Ex: 'ECG_123.png' -> '0123.png', '456' -> '0456.png'
    """
    numeros = re.findall(r'\d+', str(filename))
    if not numeros:
        return str(filename)
    numero = int(numeros[-1])
    return f"{numero:04d}.png"


def print_separador(char='=', largura=80):
    print(char * largura)


def print_secao(titulo: str):
    print_separador()
    print(f"  {titulo}")
    print_separador()


def mapear_gold_para_silver(gold_idx: np.ndarray, config: dict) -> np.ndarray:
    df_gold = pd.read_csv(config['datasets']['GOLD']['file'])
    df_silver = pd.read_csv(config['datasets']['SILVER']['file'])
    filenames = df_gold.iloc[gold_idx]['filename'].values
    fn_to_idx = {fn: i for i, fn in enumerate(df_silver['filename'].values)}
    return np.array([fn_to_idx[fn] for fn in filenames])


# # Dataset

# In[4]:


class ECGMultimodalDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        image_dir: str,
        param_cols: list,
        mode: str,
        label_col: str = 'classificacao',
        filename_col: str = 'filename',
        transform=None,
        scaler=None,
        return_filename: bool = None,
    ):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.param_cols = param_cols
        self.label_col = label_col
        self.filename_col = filename_col
        self.transform = transform
        self.scaler = scaler
        self.return_filename = return_filename
        self.mode = mode

        if self.mode not in ['hybrid', 'tabular', 'image']:
            raise ValueError(
                f"mode deve ser 'hybrid', 'tabular' ou 'image'. "
                f"Recebido: {self.mode}"
            )

        for col in param_cols:
            assert col in self.df.columns, f"Coluna ausente no dataset: {col}"
        assert label_col in self.df.columns, f"Coluna alvo ausente: {label_col}"
        assert filename_col in self.df.columns, f"Coluna filename ausente: {filename_col}"

        self.params_array = self.df[param_cols].astype(np.float32).values
        self.labels_array = self.df[label_col].astype(np.long).values
        self.filenames = self.df[filename_col].values

        if self.scaler is not None:
            self.params_array = self.scaler.transform(
                self.params_array
            ).astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        sample = {}

        label = torch.tensor(self.labels_array[idx], dtype=torch.long)
        filename_norm = normalizar_nome_arquivo(self.filenames[idx])

        if self.return_filename:
            sample['filename'] = filename_norm

        if self.mode in ['hybrid', 'tabular']:
            params = torch.tensor(self.params_array[idx], dtype=torch.float32)
            sample['params'] = params

        if self.mode in ['hybrid', 'image']:
            img_path = os.path.join(self.image_dir, filename_norm)

            img = Image.open(img_path).convert('L')

            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)

            sample['image'] = img

        sample['label'] = label

        return sample

    def get_class_distribution(self) -> dict:
        unique, counts = np.unique(self.labels_array, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}

    def get_params_dataframe(self) -> dict:
        return sefl.df[self.param_cols].copy()


# # Splits estratificados

# In[5]:


def criar_splits(dataset_silver: ECGMultimodalDataset, dataset_gold: ECGMultimodalDataset, config: dict, force_recalculate: bool = True) -> tuple:
    """
    Criar os splits train e val para o dataset PRATA
    Criar o test para o dataset OURO
    """

    os.makedirs(config['splits_dir'], exist_ok=True)

    train_path = os.path.join(config['splits_dir'], 'train_path.npy')
    val_path = os.path.join(config['splits_dir'], 'val_path.npy')
    gold_test_path = os.path.join(config['splits_dir'], 'gold_test_path.npy')

    if not force_recalculate and all(os.path.isfile(p) for p in [train_path, val_path, gold_test_path]):
        print('\tSplits encontrados. Carregando...')
        train_idx = np.load(train_path)
        val_idx = np.load(val_path)
        gold_test_idx = np.load(gold_test_path)

        print(
            f"\tTrain: {len(train_idx):,} | Val: {len(val_idx):,} | Gold Test: {len(gold_test_idx):,}")

        return train_idx, val_idx, gold_test_idx

    print('\t Calculando test split estratificado do dataset ouro...')
    labels = dataset_gold.labels_array
    indices = np.arange(len(dataset_gold))

    _, gold_test_idx = train_test_split(
        indices,
        test_size=config['test_ratio'],
        stratify=labels[indices],
        random_state=config['random_seed']
    )

    gold_test_idx_silver = mapear_gold_para_silver(
        gold_idx=gold_test_idx, config=config)

    mask = np.ones(len(dataset_silver), dtype=bool)
    mask[gold_test_idx_silver] = False
    pool_idx = np.where(mask)[0]

    print('\t Calculando splits estratificados do dataset prata...')
    labels = dataset_silver.labels_array[pool_idx]

    val_ratio = config['val_ratio'] / (
        config['train_ratio'] + config['val_ratio']
    )

    train_idx, val_idx = train_test_split(
        pool_idx,
        test_size=val_ratio,
        stratify=labels,
        random_state=config['random_seed']
    )

    np.save(train_path, train_idx)
    np.save(val_path, val_idx)
    np.save(gold_test_path, gold_test_idx_silver)

    total_silver = len(dataset_silver)
    total_gold = len(dataset_gold)

    print(
        f"  Train: {len(train_idx):,} ({len(train_idx)/total_silver*100:.1f}%)")
    print(f"  Val: {len(val_idx):,} ({len(val_idx)/total_silver*100:.1f}%)")
    print(
        f"  Test Canônico (proporção do gold): {len(gold_test_idx_silver):,} ({len(gold_test_idx_silver)/total_gold*100:.1f}%)")
    print(
        f"  Test Canônico (proporção do gold com silver): {len(gold_test_idx_silver):,} ({len(gold_test_idx_silver)/total_silver*100:.1f}%)")
    print(f"  Splits salvos em: {config['splits_dir']}/")

    return train_idx, val_idx, gold_test_idx_silver


def verificar_distribuicao_splits(labels: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, gold_test_idx: np.ndarray):
    """Imprime distribuição de classes em cada split."""
    print("\n  DISTRIBUICAO DE CLASSES POR SPLIT")
    print(
        f"  {'Split':<10} {'Total':>8} {'NORMAL (0)':>14} {'ANORMAL (1)':>14} {'Ratio':>8}")
    print(f"  {'-'*58}")

    for nome, idx in [('Train', train_idx), ('Val', val_idx), ('Gold Test', gold_test_idx)]:
        y = labels[idx]
        n0 = np.sum(y == 0)
        n1 = np.sum(y == 1)
        ratio = n1 / n0 if n0 > 0 else float('inf')
        print(
            f"  {nome:<10} {len(idx):>8,} "
            f"{n0:>8,} ({n0/len(idx)*100:.1f}%) "
            f"{n1:>8,} ({n1/len(idx)*100:.1f}%) "
            f"{ratio:>6.2f}:1"
        )


# # Scaler tabular

# In[6]:


def ajustar_scaler(train_idx: np.ndarray, csv_path: str, param_cols: list) -> StandardScaler:
    params_raw = pd.read_csv(csv_path)[param_cols].values
    X_train = params_raw[train_idx]
    scaler = StandardScaler()
    scaler.fit(X_train)
    print(f"  Scaler ajustado em {len(train_idx):,} registros de treino.")
    print(f"  Media  (primeiras 3 features): {scaler.mean_[:3].round(2)}")
    print(f"  StdDev (primeiras 3 features): {scaler.scale_[:3].round(2)}")

    return scaler


# # Transforms

# In[7]:


def get_train_transform(config: dict) -> transforms.Compose:
    """
    Aplica transformações as imagens para o treino
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=config['img_channels']),
        transforms.Resize(config['img_resize']),
        transforms.RandomAffine(
            degrees=0,
            translate=(config['aug_translate'], config['aug_translate']),
            scale=(config['aug_scale_min'], config['aug_scale_max'])
        ),
        transforms.ColorJitter(
            brightness=config['aug_brightness'],
            contrast=config['aug_contrast']
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['normalize_mean'],
            std=config['normalize_std']
        )
    ])


def get_eval_transform(config: dict) -> transforms.Compose:
    """
    Aplica normalização (sem augumentation) para o validação
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=config['img_channels']),
        transforms.Resize(config['img_resize']),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['normalize_mean'],
            std=config['normalize_std']
        )
    ])


# # Dataloaders

# In[8]:


def criar_dataloaders(train_idx, val_idx, gold_test_idx, scaler, config, mode):
    train_transform = get_train_transform(config)
    eval_transform = get_eval_transform(config)

    train_dataset = ECGMultimodalDataset(
        csv_file=config['datasets']['SILVER']['file'],
        image_dir=config['image_dir'],
        param_cols=config['param_cols'],
        label_col=config['label_col'],
        filename_col=config['filename_col'],
        transform=train_transform,
        scaler=scaler,
        mode=mode
    )

    eval_dataset = ECGMultimodalDataset(
        csv_file=config['datasets']['SILVER']['file'],
        image_dir=config['image_dir'],
        param_cols=config['param_cols'],
        label_col=config['label_col'],
        filename_col=config['filename_col'],
        transform=eval_transform,
        scaler=scaler,
        mode=mode
    )

    loader_kwargs = dict(
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        persistent_workers=config['persistent_workers'],
    )

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        shuffle=True,
        drop_last=True,
        **loader_kwargs
    )
    val_loader = DataLoader(
        Subset(eval_dataset, val_idx),
        shuffle=False,
        **loader_kwargs
    )
    gold_test_loader = DataLoader(
        Subset(eval_dataset, gold_test_idx),
        shuffle=False,
        **loader_kwargs
    )

    print(f"\tDataloaders com imagens redimensionadas")
    return train_loader, val_loader, gold_test_loader


# # Sanit test

# In[528]:


def teste_sanidade_dataset(dataset: ECGMultimodalDataset, config: dict, test_gold: bool = False) -> bool:
    """
    Executa verificações de sanidade no Dataset.
    Retorna True se todas as verificações passarem.

    Os testes de shape e range de pixels são executados sobre uma amostra
    obtida com o eval_transform aplicado explicitamente, pois o dataset
    recebido (dataset_base) não carrega transform — ele é usado apenas para
    splits e scaler. Testar shapes sobre dataset_base sem transform retornaria
    as dimensões originais da imagem (1793 x 3385), não o shape pós-processamento
    esperado pelo modelo (272 x 512).
    """
    print("\n  TESTE DE SANIDADE - DATASET")
    erros = []

    n = len(dataset)
    print(f"  [1] Tamanho do dataset        : {n:,}")
    if n == 0:
        erros.append("Dataset vazio")

    # Instanciar dataset auxiliar com eval_transform para testes de shape/range
    # O dataset_base não tem transform — os shapes testados aqui devem refletir
    # o tensor que o modelo receberá efetivamente durante treino/avaliação.

    if (test_gold):
        dataset_com_transform = ECGMultimodalDataset(
            csv_file=config['datasets']['GOLD']['file'],
            image_dir=config['image_dir'],
            param_cols=config['param_cols'],
            label_col=config['label_col'],
            filename_col=config['filename_col'],
            transform=get_eval_transform(config),
            scaler=None,  # sem scaler para isolar o teste de shape/range
            mode='hybrid'
        )
    else:
        dataset_com_transform = ECGMultimodalDataset(
            csv_file=config['datasets']['SILVER']['file'],
            image_dir=config['image_dir'],
            param_cols=config['param_cols'],
            label_col=config['label_col'],
            filename_col=config['filename_col'],
            transform=get_eval_transform(config),
            scaler=None,  # sem scaler para isolar o teste de shape/range
            mode='hybrid'
        )

    # Teste 2: shapes e label da primeira amostra (com transform)
    sample = None
    try:
        sample = dataset_com_transform[0]
        img_shape = tuple(sample['image'].shape)
        param_shape = tuple(sample['params'].shape)
        label_val = sample['label'].item()

        expected_img = (config['img_channels'],) + tuple(config['img_resize'])
        print(f"  [2] Shape imagem (com transform): {img_shape}  "
              f"(esperado: {expected_img})")
        print(f"  [3] Shape parametros            : {param_shape}  "
              f"(esperado: ({len(config['param_cols'])},))")
        print(f"  [4] Label 1a amostra            : {label_val}  "
              f"(esperado: 0 ou 1)")

        if img_shape != expected_img:
            erros.append(
                f"Shape de imagem incorreto: {img_shape} != {expected_img}")
        if param_shape != (len(config['param_cols']),):
            erros.append(f"Shape de params incorreto: {param_shape}")
        if label_val not in [0, 1]:
            erros.append(f"Label invalido: {label_val}")

    except Exception as e:
        erros.append(f"Erro ao acessar amostra 0 (com transform): {e}")

    # Teste 3: dtypes
    if sample is not None:
        try:
            assert sample['image'].dtype == torch.float32, "Imagem deve ser float32"
            assert sample['params'].dtype == torch.float32, "Params devem ser float32"
            assert sample['label'].dtype == torch.long,    "Label deve ser long"
            print(f"  [5] Dtypes                      : "
                  f"float32 / float32 / long  [OK]")
        except AssertionError as e:
            erros.append(str(e))

        # Teste 4: range de pixels após Normalize(mean=0.5, std=0.5)
        # Resultado esperado: ~[-1.0, 1.0] (pixels brancos ~+1, escuros ~-1)
        try:
            img_min = sample['image'].min().item()
            img_max = sample['image'].max().item()
            print(
                f"  [6] Range pixels (normalizado)  : [{img_min:.3f}, {img_max:.3f}]"
                f"  (esperado: ~[-1.0, 1.0])"
            )
            if img_min < -1.5 or img_max > 1.5:
                erros.append(
                    f"Range de pixels fora do esperado: [{img_min:.3f}, {img_max:.3f}]"
                )
        except Exception as e:
            erros.append(f"Erro ao verificar range: {e}")

    # Teste 5: distribuição de classes (sobre dataset_base — fonte de verdade)
    dist = dataset.get_class_distribution()
    total = sum(dist.values())
    print(
        f"  [7] Distribuicao classes        : "
        f"NORMAL={dist.get(0, 0):,} ({dist.get(0, 0)/total*100:.1f}%) | "
        f"ANORMAL={dist.get(1, 0):,} ({dist.get(1, 0)/total*100:.1f}%)"
    )

    if erros:
        print(f"\n  FALHAS DETECTADAS ({len(erros)}):")
        for e in erros:
            print(f"    - {e}")
        return False
    else:
        print(f"\n  Todos os testes de sanidade passaram.")
        return True


def teste_sanidade_dataloader(
    loader: DataLoader, nome: str, config: dict
) -> float:
    """
    Testa velocidade e consistência de um DataLoader.
    Retorna tempo médio por batch em segundos.
    """
    print(f"\n  TESTE DE SANIDADE - DATALOADER ({nome})")
    n_batches_teste = 3
    tempos = []

    try:
        for i, batch in enumerate(loader):
            if i >= n_batches_teste:
                break
            inicio = time.time()
            img = batch['image']
            params = batch['params']
            label = batch['label']
            elapsed = time.time() - inicio
            tempos.append(elapsed)

            if i == 0:
                print(f"  Batch shape - image  : {tuple(img.shape)}")
                print(f"  Batch shape - params : {tuple(params.shape)}")
                print(f"  Batch shape - label  : {tuple(label.shape)}")
                print(f"  Labels (primeiros 8) : {label[:8].tolist()}")

        media = np.mean(tempos) if tempos else 0
        print(f"  Tempo medio por batch: {media*1000:.1f} ms")
        return media

    except Exception as e:
        print(f"  ERRO no DataLoader {nome}: {e}")
        return -1.0


def verificar_data_leakage_splits(train_idx: np.ndarray, val_idx: np.ndarray, gold_test_idx_silver: np.ndarray):
    overlap_train = np.intersect1d(train_idx, gold_test_idx_silver)
    overlap_val = np.intersect1d(val_idx, gold_test_idx_silver)
    overlap_train_val = np.intersect1d(train_idx, val_idx)

    print(f"  Sobreposição train/gold_test: {len(overlap_train)} | (Espera-se 0)\n"
          f"  Sobreposição val/gold_test  : {len(overlap_val)}   | (Espera-se 0)\n"
          f"  Sobreposição train/val  : {len(overlap_train_val)} | (Espera-se 0)\n"
          )


# # Visualization

# In[529]:


def gerar_grade_amostras(
    dataset: ECGMultimodalDataset,
    config: dict,
    n_amostras: int = 8,
    visualization_gold: bool = False
):
    """Gerar grade visual com amostras balanceadas do Dataset."""
    os.makedirs(config['plots_dir'], exist_ok=True)

    # Seleciona arquivo CSV conforme flag
    csv_file = config['datasets']['GOLD']['file'] if visualization_gold else config['datasets']['SILVER']['file']
    dataset_label = 'GOLD' if visualization_gold else 'SILVER'

    # Dataset sem scaler para visualização com valores reais
    dataset_vis = ECGMultimodalDataset(
        csv_file=csv_file,
        image_dir=config['image_dir'],
        param_cols=config['param_cols'],
        label_col=config['label_col'],
        filename_col=config['filename_col'],
        transform=get_eval_transform(config),
        scaler=None,
        return_filename=True,
        mode='hybrid'
    )

    if hasattr(dataset_vis, 'df'):
        dataset_vis.df = dataset_vis.df.reset_index(drop=True)
        labels = dataset_vis.df[config['label_col']].values  # recalcula labels
    else:
        labels = dataset_vis.labels_array

    # Selecionar amostras balanceadas (N/2 de cada classe)
    labels = dataset_vis.labels_array
    idx_normal = np.where(labels == 0)[0]
    idx_anormal = np.where(labels == 1)[0]

    n_cada = n_amostras // 2
    rng = np.random.default_rng(config['random_seed'])
    sel_normal = rng.choice(idx_normal,  min(
        n_cada, len(idx_normal)),  replace=False)
    sel_anormal = rng.choice(idx_anormal, min(
        n_cada, len(idx_anormal)), replace=False)
    indices = np.concatenate([sel_normal, sel_anormal])
    rng.shuffle(indices)

    n_cols = 4
    n_rows = math.ceil(len(indices) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    fig.suptitle(
        'Amostras do ECGMultimodalDataset\n'
        f'(resize: {config["img_resize"][1]}x{config["img_resize"][0]} px | '
        f'dataset: {dataset_label} | n={len(dataset_vis):,})',
        fontsize=12, fontweight='bold', y=1.02
    )

    axes_flat = np.array(axes).flatten()

    for i, idx in enumerate(indices):
        ax = axes_flat[i]

        # Carrega amostra — __getitem__ retorna dict
        try:
            sample = dataset_vis[int(idx)]
        except Exception as e:
            print(
                f"  [AVISO] Falha ao carregar idx={idx}: {type(e).__name__}: {e}")
            ax.axis('off')
            continue

        # Extrai tensores
        img_tensor = sample.get('image')
        label_tensor = sample.get('label')
        filename = sample.get('filename', f'idx_{idx}')

        if img_tensor is None or label_tensor is None:
            print(
                f"  [AVISO] Chaves ausentes no sample idx={idx}: {list(sample.keys())}")
            ax.axis('off')
            continue

        # (1, H, W) → (H, W)
        img_np = img_tensor.squeeze(0).numpy()

        # Desnormalizar: Normalize(mean=0.5, std=0.5) → [-1,1] para [0,1]
        img_np = img_np * 0.5 + 0.5
        img_np = np.clip(img_np, 0, 1)

        label = int(label_tensor.item())
        cor_borda = '#2ecc71' if label == 0 else '#e74c3c'
        classe = 'NORMAL' if label == 0 else 'ANORMAL'

        ax.imshow(img_np, cmap='gray', aspect='auto')
        ax.set_title(f"{filename}\n{classe}", fontsize=8,
                     color=cor_borda, fontweight='bold')
        ax.axis('off')

        # Borda colorida por classe
        for spine in ax.spines.values():
            spine.set_edgecolor(cor_borda)
            spine.set_linewidth(2.5)
            spine.set_visible(True)

    # Desativa eixos vazios
    for j in range(len(indices), len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()

    output_path = os.path.join(config['plots_dir'], 'dataset_sample_grid.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Grade de amostras exportada: {output_path}")


# # Export report

# In[530]:


def exportar_relatorio_sanidade(
    resultados: dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    gold_test_idx: np.ndarray,
    config: dict
):
    """Exporta relatório de sanidade em arquivo texto."""
    os.makedirs(config['output_dir'], exist_ok=True)
    output_path = os.path.join(
        config['output_dir'], 'dataloader_sanity_report.txt')

    total = len(train_idx) + len(val_idx) + len(gold_test_idx)
    linhas = [
        "=" * 70,
        " RELATORIO DE SANIDADE DO DATASET MULTIMODAL",
        "=" * 70,
        f"Projeto     : {config['project']}",
        f"Autor       : {config['author']}",
        f"Instituicao : {config['institution']}",
        f"Data/Hora   : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
        "",
        "CONFIGURACAO DO DATASET",
        "-" * 40,
        f"Dataset         : {os.path.exists(config['datasets']['SILVER']['file'])}",
        f"Imagens         : {config['image_dir']}",
        f"N features      : {len(config['param_cols'])}",
        f"Features        : {', '.join(config['param_cols'])}",
        f"Img resize      : {config['img_resize'][1]}x{config['img_resize'][0]} px "
        f"(largura x altura) — aspect ratio 3385:1793 preservado",
        f"Img channels    : {config['img_channels']} (Gray)",
        f"Img modo orig.  : L (monocromatico, 300 DPI)",
        f"Batch size      : {config['batch_size']}",
        f"Num workers     : {config['num_workers']}",
        "",
        "SPLITS (base: GOLD, 0% imputacao)",
        "-" * 40,
        f"Total           : {total:,} registros",
        f"Train (70%)     : {len(train_idx):,} registros",
        f"Val   (15%)     : {len(val_idx):,} registros",
        f"Gold Test (15%) : {len(gold_test_idx):,} registros",
        f"Seed            : {config['random_seed']}",
        "",
        "RESULTADOS DE SANIDADE",
        "-" * 40,
        f"Dataset OK             : {resultados.get('dataset_ok', 'N/A')}",
        f"Train loader (ms/batch): {resultados.get('train_ms', -1)*1000:.1f}",
        f"Val   loader (ms/batch): {resultados.get('val_ms',   -1)*1000:.1f}",
        f"Test  loader (ms/batch): {resultados.get('test_ms',  -1)*1000:.1f}",
        "",
        "=" * 70,
    ]

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(linhas))

    print(f"  Relatorio exportado: {output_path}")


# # Main

# In[ ]:


def main(config: dict):
    print_separador()
    print(" SCRIPT 08: PYTORCH DATASET E DATALOADERS MULTIMODAIS")
    print(f"  {config['project']}")
    print(f"  Data   : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print_separador()

    # Verificar existência de paths obrigatórios
    if not os.path.exists(config['datasets']['SILVER']['file']) or not os.path.exists(config['datasets']['GOLD']['file']):
        print(
            f"\n  ERRO: Dataset nao encontrado: {os.path.exists(config['datasets']['SILVER']['file'])}")
        return
    if not os.path.exists(config['image_dir']):
        print(
            f"\n  ERRO: Diretorio de imagens nao encontrado: {config['image_dir']}")
        return

    device = get_device()
    config = patch_config_for_device(config, device)


    print_secao("1. CARREGANDO DATASET BASE (SILVER)")
    dataset_base = ECGMultimodalDataset(
        csv_file=config['datasets']['SILVER']['file'],
        image_dir=config['image_dir'],
        param_cols=config['param_cols'],
        label_col=config['label_col'],
        filename_col=config['filename_col'],
        transform=None,
        scaler=None,
        mode='hybrid'
    )
    print(f"  Dataset        : {config['datasets']['SILVER']['file']}")
    print(f"  Total registros: {len(dataset_base):,}")
    dist = dataset_base.get_class_distribution()
    total = len(dataset_base)
    print(
        f"  NORMAL  (0)    : {dist.get(0, 0):,} ({dist.get(0, 0)/total*100:.1f}%)")
    print(
        f"  ANORMAL (1)    : {dist.get(1, 0):,} ({dist.get(1, 0)/total*100:.1f}%)")
    print(f"  Img resize     : {config['img_resize'][1]}x{config['img_resize'][0]} px "
          f"(aspect ratio 3385:1793 preservado)")

    dataset_gold = ECGMultimodalDataset(
        csv_file=config['datasets']['GOLD']['file'],
        image_dir=config['image_dir'],
        param_cols=config['param_cols'],
        label_col=config['label_col'],
        filename_col=config['filename_col'],
        transform=None,
        scaler=None,
        mode='hybrid'
    )

    print_secao("2. SPLITS ESTRATIFICADOS")
    train_idx, val_idx, gold_test_idx_silver = criar_splits(
        dataset_base, dataset_gold, config, force_recalculate=True)
    verificar_distribuicao_splits(
        dataset_base.labels_array, train_idx, val_idx, gold_test_idx_silver
    )

    print_secao("3. NORMALIZACAO TABULAR (StandardScaler — fit apenas no TRAIN)")
    scaler = ajustar_scaler(
        train_idx, config['datasets']['SILVER']['file'], config['param_cols'])

    print_secao("4. CRIANDO DATALOADERS")
    train_loader, val_loader, gold_test_loader = criar_dataloaders(
        train_idx, val_idx, gold_test_idx_silver, scaler, config, 'hybrid'
    )

    print(f"  Train loader   : {len(train_loader):,} batches "
          f"(batch={config['batch_size']}, drop_last=True)")
    print(f"  Val   loader   : {len(val_loader):,} batches")
    print(f"  Gold Test  loader   : {len(gold_test_loader):,} batches")

    print_secao("5. TESTES DE SANIDADE")
    dataset_ok = teste_sanidade_dataset(dataset_base, config)
    train_ms = teste_sanidade_dataloader(train_loader, 'TRAIN', config)
    val_ms = teste_sanidade_dataloader(val_loader,   'VAL',   config)
    gold_test_ms = teste_sanidade_dataloader(
        gold_test_loader, 'GOLD_TEST', config)
    data_leakage = verificar_data_leakage_splits(
        train_idx, val_idx, gold_test_idx_silver)

    resultados = {
        'dataset_ok': dataset_ok,
        'train_ms': train_ms,
        'val_ms': val_ms,
        'gold_test_ms': gold_test_ms,
    }

    print_secao("6. EXPORTANDO ARTEFATOS")
    gerar_grade_amostras(dataset_base, config)
    exportar_relatorio_sanidade(
        resultados, train_idx, val_idx, gold_test_idx_silver, config)


if __name__ == '__main__':
    main(CONFIG)
