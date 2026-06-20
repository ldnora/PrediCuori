import os
import sys
import subprocess
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).parent.parent / '.env')


def setup_environment() -> dict:
    ROOT = Path(__file__).parent.parent

    if os.path.exists('/kaggle/input'):
        logger.info('Ambiente: Kaggle')
        csv_dir = '/kaggle/input/datasets/ldnora/predicuori-ecg-csv'
        image_dir = '/kaggle/input/datasets/ldnora/predicuori-image-tracings/image_tracings'
        return {
            'image_dir':  image_dir,
            'splits_dir': '/kaggle/working/repo/src/splits/',
            'output_dir': '/kaggle/working/',
            'csv_gold':   os.path.join(csv_dir, 'ecg_gold_completo_classified.csv'),
            'csv_silver': os.path.join(csv_dir, 'ecg_silver_knn_imputado_classified.csv'),
        }

    logger.info('Ambiente: Local')
    csv_dir = ROOT / 'data' / 'csv'
    return {
        'image_dir':  str(ROOT / os.getenv('LOCAL_IMAGE_DIR', 'image_tracings')),
        'splits_dir': str(ROOT / 'src' / 'splits'),
        'output_dir': str(ROOT / os.getenv('LOCAL_OUTPUT_DIR', 'resultados_e_metricas')),
        'csv_gold':   str(csv_dir / 'ecg_gold_completo_classified.csv'),
        'csv_silver': str(csv_dir / 'ecg_silver_knn_imputado_classified.csv'),
    }


_PATHS = setup_environment()

SHARED = {
    'datasets': {
        'GOLD':   {'file': _PATHS['csv_gold'],   'n': 3013, 'imputacao_pct': 0.0,  'metodo': 'Sem imputacao'},
        'SILVER': {'file': _PATHS['csv_silver'],  'n': 3481, 'imputacao_pct': 1.67, 'metodo': 'KNN'},
    },
    'image_dir':  _PATHS['image_dir'],
    'splits_dir': _PATHS['splits_dir'],
    'param_cols': [
        'HR', 'Pd', 'PR', 'QRS_Dur', 'QT', 'QTC',
        'P_axis', 'QRS_axis', 'T_axis',
        'RV5', 'SV1', 'RV5_SV1_sum', 'RV6', 'SV2'
    ],
    'label_col': 'classificacao',
    'filename_col': 'filename',
    'batch_size':  32,
    'img_resize': (272, 512),
    'img_channels': 1,
    'normalize_mean': [0.5],
    'normalize_std': [0.5],
    'aug_translate': 0.02,
    'aug_scale_min': 0.98,
    'aug_scale_max': 1.02,
    'aug_brightness': 0.1,
    'aug_contrast': 0.1,
    'train_ratio': 0.70,
    'val_ratio': 0.15,
    'random_seed': 42,
    'author': 'Leandro Dalla Nora',
    'institution': 'UFSM - Departamento de Computação Aplicada - Curso de Sistemas de Informação',
    'project': 'PrediCuori',
}


def build_config_06() -> dict:
    return {
        **SHARED,
        'output_dir': os.path.join(_PATHS['output_dir'], 'script_06_modelos_tabulares'),
        'plots_dir': os.path.join(_PATHS['output_dir'], 'script_06_modelos_tabulares', 'plots_comparativos'),
        'physiological_ranges': {
            "HR": (25, 300),
            "Pd": (40, 200),
            "PR": (50, 400),
            "QRS_Dur": (40, 250),
            "QT": (200, 700),
            "QTC": (250, 700),
            "P_axis": (-90, 120),
            "QRS_axis": (-180, 180),
            "T_axis": (-180, 180),
            "RV5": (0, 15),
            "SV1": (0, 15),
            "RV5_SV1_sum": (0, 25),
            "RV6": (0, 15),
            "SV2": (0, 15)
        },
        'test_ratio': 0.15,
        'metric': "roc_auc",
        'n_trials': 1,
        'cv_folds': 2,
    }


def build_config_07() -> dict:
    return {
        **SHARED,
        'output_dir': os.path.join(_PATHS['output_dir'], 'script_07_verificar_integridade'),
        'plots_dir': os.path.join(_PATHS['output_dir'], 'script_07_verificar_integridade', 'plots_comparativos'),
        'expected_total_files': 3900,
        # Especificações confirmadas das imagens (amostra real verificada)
        'expected_size': (3385, 1793),  # (largura, altura) em pixels
        'expected_mode': 'L',           # modo Gray (monocromático)
        'expected_dpi': (300, 300),    # resolução esperada
        'min_width': 2000,          # largura mínima aceitável
        'min_height': 1000,          # altura mínima aceitável
        # Thresholds de qualidade (escala 0-255, modo L/grayscale)
        'min_brightness': 100,       # brilho médio mínimo aceitável máximo aceitável (254 = quase branco puro)
        'max_brightness': 254,
        'min_contrast': 5,       # desvio padrão mínimo de intensidade
        'max_file_size_mb': 10,       # tamanho máximo por arquivo em MB
    }


def build_config_08() -> dict:
    return {
        **SHARED,
        'output_dir': os.path.join(_PATHS['output_dir'], 'script_08_pytorch_dataset'),
        'plots_dir': os.path.join(_PATHS['output_dir'], 'script_08_pytorch_dataset', 'plots_comparativos'),
        'test_ratio': 0.173249253235977,
        'num_workers': 4,
        'pin_memory': False,
        'persistent_workers': False,
    }


def build_config_09() -> dict:
    return {
        **SHARED,
        'results_dir': os.path.join(_PATHS['output_dir'], 'script_09_modelo_hibrido'),
        'checkpoints_dir': os.path.join(_PATHS['output_dir'], 'script_09_modelo_hibrido', 'checkpoints'),
        'test_ratio': 0.173249253235977,
        'cnn_features': 1024,
        'mlp_out': 32,
        'fusion_dim': 1056,
        'n_classes': 2,
        'dropout': 0.4,
        'x': 4,
        'num_workers': 0,
        'pin_memory': False,
        'persistent_workers': False,
        'models_modes': {
            'densenet121_hybrid': 'hybrid',
            'resnet50_hybrid': 'hybrid',
            'efficientnet_b0_hybrid': 'hybrid',
            'cnn_only': 'image',
            'mlp_only': 'tabular',
        },
        'modes': ['hybrid', 'image', 'tabular'],
        'warmup_epochs': 10,
        'warmup_lr': 1e-3,
        'finetune_epochs': 40,
        'lr_cnn': 1e-5,
        'lr_mlp': 1e-4,
        'lr_fusion': 1e-4,
        'weight_decay': 1e-4,
        'lr_patience': 5,
        'lr_factor': 0.5,
        'early_stop_patience': 10,
        'grad_clip': 1.0,
        'delta_threshold': 0.02,
    }
