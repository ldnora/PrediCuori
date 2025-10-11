# PrediCuori
SISTEMA INTELIGENTE PARA PREDIÇÃO DE EVENTOS CARDIOVASCULARES AGUDOS: INTEGRAÇÃO DE ECG E DADOS CLÍNICOS USANDO INTELIGÊNCIA ARTIFICIAL

## Como usar 

Ative o venv.

`source .venv/bin/activate`

Instale as dependências

`pip install -r requirements.txt`

Rode o programa principal

`python dicom2cleanecg.py --dpi 300 300 --grayscale`

Pseudoanonimize os dados 

`python src/pseudonymization.py`


### Como remover a pseudoanonimização 
Você vai precisar da private key original.

`src/pseudonymization.py --input ./data/ecg_images/png-anonymazed --output ./data/ecg_images/teste --reverse`

