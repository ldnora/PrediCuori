import os
import sys
import subprocess
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).parent.parent / '.env')


def setup_environment() -> dict:
    if os.path.exists('/kaggle/input'):
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        token = secrets.get_secret("GITHUB_TOKEN")
        repo = secrets.get_secret("GITHUB_REPO")

        subprocess.run([
            'git', 'clone', '--quiet', '--depth', '1',
            f'https://{token}@{repo}',
            '/kaggle/working/repo'
        ], check=True)

        subprocess.run([
            'pip', 'install', '-q', '-r',
            '/kaggle/working/repo/requirements.txt'
        ], check=True)

        sys.path.insert(0, '/kaggle/working/repo/src')
        logger.info('Ambiente: Kaggle')

        dataset_csv = os.getenv('KAGGLE_DATASET_CSV').split('/')[-1]
        dataset_imgs = os.getenv('KAGGLE_DATASET_IMGS').split('/')[-1]

        return {
            'image_dir': f'/kaggle/input/{dataset_imgs}/',
            'splits_dir': '/kaggle/working/repo/src/splits/',
            'base_output': '/kaggle/working/',
            'csv_gold': f'/kaggle/input/{dataset_csv}/ecg_gold_completo_classified.csv',
            'csv_silver': f'/kaggle/input/{dataset_csv}/ecg_silver_knn_imputado_classified.csv',
        }

    ROOT = Path(__file__).parent.parent
    logger.info('Ambiente: Local')

    return {
        'image_dir': str(ROOT / os.getenv('LOCAL_IMAGE_DIR', 'image_tracings')),
        'splits_dir': str(ROOT / 'src' / 'splits'),
        'base_output str': (ROOT / os.getenv('LOCAL_OUTPUT_DIR', 'resultados_e_metricas')),
        'csv_gold': str(ROOT / 'data' / 'csv' / 'ecg_gold_completo_classified.csv'),
        'csv_silver': str(ROOT / 'data' / 'csv' / 'ecg_silver_knn_imputado_classified.csv'),
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
    'img_resize': (136, 256),
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
    'test_ratio': 0.173249253235977,
    'random_seed': 42,
    'author': 'Leandro Dalla Nora',
    'institution': 'UFSM - Departamento de Computação Aplicada - Curso de Sistemas de Informação',
    'project': 'PrediCuori',
}


def build_config_08() -> dict:
    return {
        **SHARED,
        'output_dir': os.path.join(_PATHS['base_output'], 'script_08_pytorch_dataset'),
        'plots_dir': os.path.join(_PATHS['base_output'], 'script_08_pytorch_dataset', 'plots_comparativos'),
        'batch_size': 32,
        'num_workers': 4,
        'pin_memory': False,
        'persistent_workers': False,
    }


def build_config_09() -> dict:
    return {
        **SHARED,
        'results_dir': os.path.join(_PATHS['base_output'], 'script_09_modelo_hibrido'),
        'checkpoints_dir': os.path.join(_PATHS['base_output'], 'script_09_modelo_hibrido', 'checkpoints'),
        'cnn_features': 1024,
        'mlp_out': 32,
        'fusion_dim': 1056,
        'n_classes': 2,
        'dropout': 0.4,
        'batch_size': 4,
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
        'warmup_epochs': 1,
        'warmup_lr': 1e-3,
        'finetune_epochs': 1,
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
