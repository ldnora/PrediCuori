import json
import subprocess
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent / '.env')

KAGGLE_USER = os.getenv('KAGGLE_USERNAME')
KERNEL_SLUG = os.getenv('KERNEL_SLUG')
GITHUB_REPO = os.getenv('GITHUB_REPO')
DATASET_CSV = os.getenv('KAGGLE_DATASET_CSV')
DATASET_IMGS = os.getenv('KAGGLE_DATASET_IMGS')

notebook = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4,
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import subprocess, sys\n",
                "from kaggle_secrets import UserSecretsClient\n",
                "\n",
                "token = UserSecretsClient().get_secret('GITHUB_TOKEN')\n",
                f"subprocess.run([\n",
                f"    'git', 'clone', '--quiet', '--depth', '1',\n",
                f"    f'https://{{token}}@{GITHUB_REPO}',\n",
                f"    '/kaggle/working/repo'\n",
                f"], check=True)\n",
                "\n",
                "# instala dependências do repo\n",
                "subprocess.run(['pip', 'install', '-q', '-r',\n",
                "    '/kaggle/working/repo/requirements.txt'], check=True)\n",
                "\n",
                "sys.path.insert(0, '/kaggle/working/repo/src')\n",
                "\n",
                "from script_09_modelo_hibrido import main\n",
                "main()\n"
            ]
        }
    ]
}

with open("kernel.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

metadata = {
    "id": f"{KAGGLE_USER}/{KERNEL_SLUG}",
    "title": KERNEL_SLUG,
    "code_file": "kernel.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": True,
    "enable_internet": True,
    "dataset_sources": [DATASET_CSV, DATASET_IMGS],
    "kernel_sources": []
}

with open("kernel-metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

subprocess.run(["kaggle", "kernels", "push", "-p", "."], check=True)

print(f"\nKernel submetido!")
print(f"Status : kaggle kernels status {KAGGLE_USER}/{KERNEL_SLUG}")
print(f"Logs   : kaggle kernels output {KAGGLE_USER}/{KERNEL_SLUG}")
print(f"URL    : https://www.kaggle.com/code/{KAGGLE_USER}/{KERNEL_SLUG}")