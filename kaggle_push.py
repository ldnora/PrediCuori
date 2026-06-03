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
                "import subprocess, sys, os, shutil\n",
                "from kaggle_secrets import UserSecretsClient\n",
                "\n",
                "# clone único — antes de qualquer import\n",
                "repo_path = '/kaggle/working/repo'\n",
                "if not os.path.exists(repo_path):\n",
                "    token = UserSecretsClient().get_secret('GITHUB_TOKEN')\n",
                "    repo  = UserSecretsClient().get_secret('GITHUB_REPO')\n",
                "    subprocess.run([\n",
                "        'git', 'clone', '--quiet', '--depth', '1',\n",
                "        f'https://{token}@{repo}',\n",
                "        repo_path\n",
                "    ], check=True)\n",
                "    print('Clone OK')\n",
                "else:\n",
                "    print('Repo já existe, pulando clone')\n",
                "\n",
                "sys.path.insert(0, f'{repo_path}/src')\n",
                "\n",
                "from script_09_modelo_hibrido import main\n",
                "main()\n",
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
