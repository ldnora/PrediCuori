# PrediCuori


PrediCuori: sistema inteligente para predição de eventos cardiovasculares anormais: integração de ecg e dados clínicos usando inteligência artificial.



## Boas-vindas

Olá! Espero que você, querido leitor, esteja bem. 

Eu sou o Leandro Dalla Nora e quero dar as boas-vindas ao meu Trabalho de Conclusão de Curso do Bacharelado em Sistemas de Informação pela Universidade Federal de Santa Maria. 




## Objetivo 


## Objetivo deste repositório

O objetivo deste repositório é o versionamento e armazenamento do código desenvolvido para a minha tese de gradução; disponibilidade de todos os artefatos criados e gerados, para que os experimentos sejam reproduzidos por outros pesquisadores; 



## Objetivo do trabalho


### Objetivo geral

Criar um modelo multimodal, que integre as 12 derivações dos ECGs com os dados tabulares extraídos do mesmo, para identificar anormalidades cardiovasculares dos pacientes do HUSM.


### Objetivo específico

- Criar um banco de dados anonimizado e aberto, composto por dados tabulares e imagens dos traçados dos eletrocardiogramas extraídos do Hospital Universitário de Santa Maria (HUSM);
- Criar e treinar um modelo tabular de Machine Learning;
- Construir e treinar um modelo multimodal, com base nos dados tabulares e nas imagens dos traçados dos ECGs.


## Descrição do repositório

```bash
.
├── devbox.json
├── kaggle_push.py
├── README.md
├── requirements.txt
├── resultados_e_metricas
│   ├── _20260416_153715
│   │   ├── hyperparams.json
│   │   ├── metrics.csv
│   │   └── models
│   │       ├── gradient_boosting.pkl
│   │       ├── logistic_regression.pkl
│   │       ├── mlp.pkl
│   │       ├── random_forest.pkl
│   │       ├── svm.pkl
│   │       └── xgboost.pkl
│   ├── outliers_stats.txt
│   ├── plots_comparativos
│   │   ├── baseline_comparison.png
│   │   ├── bell_chart_prata.png
│   │   ├── confusion_matrix_gradient_boosting.png
│   │   ├── confusion_matrix_logistic_regression.png
│   │   ├── confusion_matrix_mlp.png
│   │   ├── confusion_matrix_random_forest.png
│   │   ├── confusion_matrix_svm.png
│   │   ├── confusion_matrix_xgboost.png
│   │   ├── confusion_matrix_XGB.png
│   │   ├── feature_importance_gradient_boosting.png
│   │   ├── feature_importance_logistic_regression.png
│   │   ├── feature_importance_random_forest.png
│   │   ├── feature_importance_RF.png
│   │   ├── feature_importance_xgboost.png
│   │   ├── feature_importance_XGB.png
│   │   ├── histogramas_sv2_rv6.png
│   │   └── roc_curves_comparison.png
│   ├── script_07_verificar_integridade
│   │   ├── integrity_report.csv
│   │   ├── integrity_summary.txt
│   │   ├── log.txt
│   │   └── plots_comparativos
│   │       └── image_quality_stats.png
│   ├── script_08_pytorch_dataset
│   │   ├── dataloader_sanity_report.txt
│   │   ├── log.txt
│   │   └── plots_comparativos
│   │       └── dataset_sample_grid.png
│   └── script_09_modelo_hibrido
│       ├── checkpoints
│       │   ├── cnn_only
│       │   │   ├── best_finetune.pt
│       │   │   ├── best_warmup.pt
│       │   │   └── modelo_cnn_only_final.pt
│       │   ├── densenet121_hybrid
│       │   │   ├── best_finetune.pt
│       │   │   ├── best_warmup.pt
│       │   │   └── modelo_densenet121_hybrid_final.pt
│       │   ├── efficientnet_b0_hybrid
│       │   │   ├── best_finetune.pt
│       │   │   ├── best_warmup.pt
│       │   │   └── modelo_efficientnet_b0_hybrid_final.pt
│       │   ├── mlp_only
│       │   │   ├── best_finetune.pt
│       │   │   └── modelo_mlp_only_final.pt
│       │   └── resnet50_hybrid
│       │       ├── best_finetune.pt
│       │       ├── best_warmup.pt
│       │       └── modelo_resnet50_hybrid_final.pt
│       ├── comparative_results.csv
│       ├── comparative_results.json
│       ├── historico_cnn_only.json
│       ├── historico_densenet121_hybrid.json
│       ├── historico_efficientnet_b0_hybrid.json
│       ├── historico_mlp_only.json
│       ├── historico_resnet50_hybrid.json
│       ├── plots_comparativos
│       │   ├── comparative_auc_barras.png
│       │   └── comparative_roc.png
│       ├── predicuori.log
│       └── script_09_MAIN_20260608_201223.log
└── src
    ├── config.py
    ├── device_utils.py
    ├── script_06_baseline_tabular_v2.ipynb
    ├── script_07_verificar_integridade_imagens.py
    ├── script_08_pytorch_dataset.py
    ├── script_09_modelo_hibrido.py
    └── splits
        ├── gold_test_indices.npy
        ├── train_indices.npy
        └── val_indices.npy
```

O arquivo `devbox.json` apresenta a configuração de sheel DevBox, a qual permite criar um ambiente sheel isolado.

O `kaggle_push.py` é o arquivo que apresenta a configuração de infraestrutura utilizada para o treinamento dos modelos multimodais no hardware disponibilizado pelo Kaggle (NVIDIA TESLA P100 GPU). Ele é responsável por criar um notebook (`.ipynb`) localmente, enviar ele ao Kaggle, sincronizar o código com o GitHub e executá-lo.

O `requirements.txt` apresenta a lista de bibliotecas utilizadas nos códigos em python.

A pasta `resultados_e_metricas` apresenta todos os resultados coletados durante a execução dos códigos. Estes resultados são: modelos treinados, plots, métricas de cada modelo, reports de integridade dos datasets e logs de execução. Cada subpasta refere-se ao seu respectivo scritp. Por exemplo, a subpasta `script_07_verificar_integridade` se refere ao código `src/script_07_verificar_integridade_imagens.py`. 

A pasta `src` apresenta todos os códigos desenvolvidos dos modelos e testes de integridade.

O arquivo `src/config.py` apresenta um dicionário com a configuração usada em cada script. Essa configuração contém paths dos csvs, imagens e variáveis. A configuração dos paths é feita pela detecção do ambiente o qual o código está sendo executado, seja ele em um computador ou em uma outra infraestrutura, como a do Kaggle que foi utilizada.

O arquivo `src/device_utils.py` detecta qual é o hardware que está sendo utilizado para executar o código. Ele é usado para o `src/script_09_modelo_hibrido.py`, pois este exige hardware mais potente.

O arquivo `src/`

O script `src/script_07_verificar_integridade_imagens.py` é responsável por verificar a integridade das imagens.

O script `src/script_08_pytorch_dataset.py` é responsável por criar os índices dos splits, dataloaders, pytorch dataset e aplicar transformações para as imagens.

O script `src/script_09_modelo_hibrido.py` contém os modelos multimodais utilizados neste trabalho.



## Como executar?

"Hmm, achei muito interessante esse trabalho, mas como fazer os meus próprios experimentos?"

Calma jovem, vou lhe explicar:


#### Configuração do ambiente

Você pode configurar o seu ambiente de duas formas:

###### DevBox + venv 

Você pode instalar o DevBox em sua máquina. Há [esse tutorial oficial](https://www.jetify.com/docs/devbox/installing-devbox) da ferramenta. 


Execute o comando `devbox shell` no terminal. Com ele, toda a configuração de ambiente será feita.


###### Venv

Crie um ambiente virtual: `python -m venv .venv`

Ative o ambiente virtual para o Windows: `.venv\Scripts\activate`

Ative o ambiente virtual para o Gnu Linux/MacOS: `.venv\bin/activate`


#### Instalação dos datasets

TODO


#### Execução

Rode no terminal os comandos: 

Scritp 07: `python src/script_07_verificar_integridade_imagens.py`

Script 08: `python src/script_08_pytorch_dataset.py`

Script 09: 

Para em sua máquina: `python src/script_09_modelo_hibrido.py`

Para executar na infraestrutura do Kaggle: `python kaggle_push.py` 

> Observação:
> Para que a execução ocorra no Kaggle, é necessário configurar GITHUB_TOKEN, GITHUB_REPO, KAGGLE_USERNAME, KAGGLE_KEY, KAGGLE_DATASET_CSV e KAGGLE_DATASET_IMGS. Use o `.env.example` como base.