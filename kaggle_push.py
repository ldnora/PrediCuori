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
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')

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
                "import subprocess, sys, os\n\n",

                "subprocess.run(['pip', 'install', '-q',\n",
                "    'torch==2.5.1', 'torchvision==0.20.1',\n",
                "    '--index-url', 'https://download.pytorch.org/whl/cu124'\n",
                "], check=True)\n",

                "import IPython\n",
                "IPython.Application.instance().kernel.do_shutdown(True)\n",

                f"token = '{GITHUB_TOKEN}'\n",
                f"repo  = '{GITHUB_REPO}'\n",
                "\n",
                "repo_path = '/kaggle/working/repo'\n",
                "if not os.path.exists(repo_path):\n",
                "    subprocess.run([\n",
                "        'git', 'clone', '--quiet', '--depth', '1',\n",
                "        f'https://{token}@{repo}',\n",
                "        repo_path\n",
                "    ], check=True)\n",
                "    print('Clone OK')\n",
                "else:\n",
                "    subprocess.run(['git', 'pull', '--quiet'], cwd=repo_path, check=True)\n",
                "    print('Pull OK')\n",
                "\n",
                "for mod in list(sys.modules.keys()):\n",
                "    if 'config' in mod or 'script_09' in mod:\n",
                "        del sys.modules[mod]\n",
                "\n",
                "sys.path.insert(0, f'{repo_path}/src')\n",
                "\n",
                "os.environ['CUDA_LAUNCH_BLOCKING'] = '1'\n",
                "os.environ['TORCH_USE_CUDA_DSA'] = '1'\n",

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
    "enable_gpu": True,
    "enable_tpu": False,
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

