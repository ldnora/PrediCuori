#!/usr/bin/env python
# coding: utf-8

# # Verificação de Integridade das Imagens
# 
# 
# ## Objetivo e Função no Pipeline
# 
# O scritp_07_verificar_integridade_imagens.ipynb realiza uma auditoria completa e independente do corpus físico
# de imagens, iterando sobre todos os arquivos PNG presentes em image_tracings/. Esta etapa é obrigatória antes
# do treinamento pois garante que nenhuma imagem corrompida, com resolução incorreta ou com problemas de
# qualidade visual seja incluída nos DataLoaders.
# 
# O script opera em duas fases: (1) auditoria física independente dos CSVs e (2) cruzamento de cobertura com os
# CSVs disponíveis para verificar quais pacientes possuem registro tabular.
# 
# 
# ## O Que o Script Verifica
# 
# - Corrupção de arquivo (tentativa de abertura e leitura de cada PNG via Pillow)
# - Resolução esperada: 3385 × 1793 pixels (mínimo aceitável: 2000 × 1000)
# - Modo de cor: Grayscale (L) — rejeita imagens RGB, RGBA ou outras
# - Brilho médio: entre 100 e 254 (escala 0-255) — rejeita imagens muito escuras ou saturadas
# - Contraste: desvio padrão mínimo de 5 — rejeita imagens planas (sem informação)
# - Tamanho máximo: 10 MB por arquivo
# 
# ## Adaptação para o Escopo Silver KNN
# O script_07 itera sobre os 3.900 arquivos físicos do diretório — esta parte não precisa ser modificada. Porém, a
# seção de cruzamento de cobertura (etapa pós-auditoria) pode ser simplificada para inspecionar apenas o CSV do
# Silver KNN:
# 
# ```python 
# # Em CONFIG (scritp_07_verificar_integridade_imagens.ipynb)
# # Manter apenas GOLD e SILVER_KNN no dicionário csv_files:
# 'csv_files': {
# 'GOLD' : 'ecg_gold_completo_classified.csv',
# 'SILVER_KNN': 'ecg_silver_knn_imputado_classified.csv',
# # BRONZE_HYBRID, BRONZE_MICE, BRONZE_KNN — comentar ou remover
# },
# ```
# 
# 
# > Impacto da simplificação:
# > 
# > Remover os datasets BRONZE do dicionário csv_files não afeta a auditoria física (que é independente de CSV). Apenas o relatório de cobertura ficará mais enxuto, listando somente GOLD e SILVER_KNN. A integridade das imagens é auditada integralmente de qualquer forma.
# 
# ##  Execução
# Execute cada uma das células de código ou todas elas.
# 
# Tempo estimado: 10–30 minutos (depende da velocidade de I/O do disco e da presença de GPU/CPU para cálculos
# de numpy). Monitorar o console para mensagens de WARNING ou CORRUPTED.
# 
# ## Artefatos Gerados
# 
# | Artefato                | Local                   | Conteúdo                                                                 |
# |-------------------------|-------------------------|---------------------------------------------------------------------------|
# | integrity_report.csv    | ../resultados_e_metricas/script_07_verificar_integridade/ | Registro por imagem: status, resolução, brilho, contraste                |
# | integrity_summary.txt   | ../resultados_e_metricas/script_07_verificar_integridade/ | Sumário textual com taxa de integridade e cobertura por dataset          |
# | image_quality_stats.png | ../resultados_e_metricas/script_07_verificar_integridade/plots_comparativos/    | Painel de 6 gráficos: status, brilho, contraste, scatter, tamanho        |
# 
# ## Critérios de Aprovação
# Antes de prosseguir para o script_08, verifique no integrity_summary.txt:
# - Taxa de integridade geral >= 99,5% (aceitável até 20 imagens problemáticas)
# - Imagens corrompidas: 0 (ideal) — se > 0, investigar individualmente
# - Cobertura SILVER_KNN: todos os 3.481 registros devem ter imagem correspondente
# - Se houver imagens sem cobertura no SILVER_KNN, anotar os filenames ausentes

# In[47]:


import os
import re
import csv
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')


# In[48]:


CONFIG = {
    # Diretório físico de imagens — fonte primária da auditoria
    'image_dir':   'data/image_tracings',
    'output_dir':  '/resultados_e_metricas/script_07_verificar_integridade/',
    'plots_dir':   '/resultados_e_metricas/script_07_verificar_integridade/plots_comparativos/',

    # CSVs para cruzamento de cobertura (pós-auditoria)
    # O Bronze não será usado nesse contexto
    'datasets_dir': '/data/csv',
    'csv_files': {
        'GOLD':         'ecg_gold_completo_classified.csv',
        'SILVER_KNN':   'ecg_silver_knn_imputado_classified.csv',
        # 'BRONZE_HYBRID':'ecg_bronze_hybrid_imputado_classified.csv',
        # 'BRONZE_MICE':  'ecg_bronze_mice_imputado_classified.csv',
        # 'BRONZE_KNN':   'ecg_bronze_knn_imputado_classified.csv',
    },
    'filename_col': 'filename',

    # Contagem esperada do corpus físico
    'expected_total_files': 3900,

    # Especificações confirmadas das imagens (amostra real verificada)
    'expected_size': (3385, 1793),  # (largura, altura) em pixels
    'expected_mode': 'L',           # modo Gray (monocromático)
    'expected_dpi':  (300, 300),    # resolução esperada
    'min_width':     2000,          # largura mínima aceitável
    'min_height':    1000,          # altura mínima aceitável

    # Thresholds de qualidade (escala 0-255, modo L/grayscale)
    'min_brightness':    100,       # brilho médio mínimo aceitável
    'max_brightness':    254,       # máximo aceitável (254 = quase branco puro)
    'min_contrast':        5,       # desvio padrão mínimo de intensidade
    'max_file_size_mb':   10,       # tamanho máximo por arquivo em MB

    # Metadados
    'author':         "Leandro Dalla Nora",
    'institution':    'UFSM - Departamento de Computação Aplicada - Curso de Sistemas de Informação',
    'project':        'PrediCuori'
}


# # Funções utilitárias

# In[49]:


def normalizar_nome_arquivo(filename: str) -> str:
    """
    Normaliza filename para padrão zero-padded .png.
    Ex: 'ECG_123.png' -> '0123.png'
    """
    numeros = re.findall(r'\d+', str(filename))
    if not numeros:
        return str(filename)
    return f"{int(numeros[-1]):04d}.png"


def formatar_tamanho(bytes_size: int) -> str:
    """Formata tamanho em bytes para string legível."""
    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024 ** 2:
        return f"{bytes_size / 1024:.1f} KB"
    return f"{bytes_size / (1024 ** 2):.2f} MB"


def calcular_estatisticas_imagem(img_path: str) -> dict:
    """
    Calcula estatísticas de qualidade de uma imagem PNG em modo grayscale.
    Retorna dicionário com brilho médio, contraste e indicadores de pixel.
    """
    try:
        img = Image.open(img_path).convert('L')
        arr = np.array(img, dtype=np.float32)
        return {
            'brightness_mean': float(np.mean(arr)),
            'brightness_std':  float(np.std(arr)),
            'contrast':        float(np.std(arr)),
            'pixel_min':       float(np.min(arr)),
            'pixel_max':       float(np.max(arr)),
            'dark_ratio':      float(np.mean(arr < 50)),
            'bright_ratio':    float(np.mean(arr > 200)),
        }
    except Exception:
        return {k: -1 for k in [
            'brightness_mean', 'brightness_std', 'contrast',
            'pixel_min', 'pixel_max', 'dark_ratio', 'bright_ratio',
        ]}


def print_separador(char='=', largura=80):
    print(char * largura)


def print_secao(titulo: str):
    print_separador()
    print(f"  {titulo}")
    print_separador()


# # Auditoria física do diretório

# In[50]:


def contagem_de_arquivos(image_dir: str, config: dict) -> tuple[list[str], int, int]:
    arquivos = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith('.png')
    ])
    qtd_imagens_encontradas = len(arquivos)
    qtd_imagens_esperadas = config["expected_total_files"]

    return arquivos, qtd_imagens_encontradas, qtd_imagens_esperadas

def verificar_tamanho_arquivo(img_path: str, config, stats: dict, record: dict) -> tuple[dict, dict]:
    file_size = os.path.getsize(img_path)
    record['file_size_bytes'] = file_size
    if file_size > config['max_file_size_mb'] * 1024 * 1024:
        record['issues'].append(f"arquivo_grande_{formatar_tamanho(file_size)}")
        stats['oversized'].append(filename)

    return stats, record

def verificar_modo_de_cor(img: Image.Image, config: dict, filename: str, stats: dict, record: dict) -> tuple[dict, dict]:
    record['mode'] = img.mode
    if img.mode != config['expected_mode']:
        record['issues'].append(f'modo_incorredo_{img.mode}')
        stats['wrong_mode'].append(filename)

    return stats, record

def verificar_qualidade(img_path: str, config: dict, stats: dict, record: dict) -> tuple[dict, dict]:
    est = calcular_estatisticas_imagem(img_path)
    record['brightness_mean'] = round(est['brightness_mean'], 2)
    record['contrast']        = round(est['contrast'], 2)

    if est['brightness_mean'] < config['min_brightness']:
        record['issues'].append(f"brilho_baixo_{est['brightness_mean']:.1f}")
        stats['low_brightness'].append(filename)
    if est['brightness_mean'] > config['max_brightness']:
        record['issues'].append(f"brilho_alto_{est['brightness_mean']:.1f}")
    if est['contrast'] < config['min_contrast']:
        record['issues'].append(f"contraste_baixo_{est['contrast']:.1f}")
        stats['low_contrast'].append(filename)

    return stats, record

def verificar_resolucao(img: Image.Image, config: dict, filename: str, stats: dict, record: dict) -> tuple[dict, dict]:
    w, h = img.size
    record['width'] = w
    record['height'] = h
    if (w, h) != tuple(config['expected_size']):
        record['issues'].append(f"resolucao_incorreta_{w}x{h}")
        stats['wrong_resolution'].append(filename)

    if w < config['min_width'] or h < config['min_height']:
        record['issues'].append(f"dimensoes_insuficientes_{w}x{h}")

    return stats, record

def auditar_corpus_fisico(image_dir: str, config: dict) -> dict:
    """
    Itera sobre TODOS os arquivos PNG presentes fisicamente em image_dir.
    Esta é a função principal da auditoria — independe de qualquer CSV.

    Retorna dicionário com stats agregados e lista detalhada por arquivo.
    """
    print_secao("AUDITORIA FÍSICA DO CORPUS DE IMAGENS")

    arquivos, qtd_imagens_encontradas, qtd_imagens_esperadas = contagem_de_arquivos(image_dir, config)

    print(f"\n  Diretório         : {image_dir}")
    print(f"  Arquivos PNG encontrados : {qtd_imagens_encontradas:,}")
    delta = qtd_imagens_encontradas - qtd_imagens_esperadas
    status = "OK" if delta == 0 else f"DIVERGÊNCIA ({delta:+d})"
    print(f"  Contagem esperada        : {qtd_imagens_esperadas:,}  [{status}]")
    print(f"  Resolução esperada       : {config['expected_size'][0]}x{config['expected_size'][1]} px")
    print(f"  Modo esperado            : {config['expected_mode']} (Gray/monocromático)\n")

    stats = {
        'qtd_imagens_encontradas':  qtd_imagens_encontradas,
        'expected_total':           qtd_imagens_esperadas,
        'verificadas':              0,
        'ok':                       0,
        'corrupted':                [],
        'wrong_resolution':         [],
        'wrong_mode':               [],
        'low_brightness':           [],
        'low_contrast':             [],
        'oversized':                [],
    }

    records = []
    inicio = time.time()
    intervalo_log = max(1, qtd_imagens_encontradas // 20)

    for i, filename in enumerate(arquivos):
        img_path = os.path.join(image_dir, filename)

        record = {
            'filename':        filename,
            'img_path':        img_path,
            'status':          'OK',
            'issues':          [],
            'width':           None,
            'height':          None,
            'mode':            None,
            'file_size_bytes': None,
            'brightness_mean': None,
            'contrast':        None,
        }

        stats, record = verificar_tamanho_arquivo(img_path, config, stats, record)

        try:
            img = Image.open(img_path)

            stats, record = verificar_modo_de_cor(img, config, filename, stats, record)

            stats, record = verificar_resolucao(img, config, filename, stats, record)

            # Integridade estrutural (detecta truncamento e corrupção de cabeçalho)
            img.verify()
            stats['verificadas'] += 1

        except UnidentifiedImageError:
            record['status'] = 'CORRUPTED'
            record['issues'].append('formato_nao_identificado')
            stats['corrupted'].append(filename)
            records.append(record)
            continue
        except Exception as e:
            record['status'] = 'CORRUPTED'
            record['issues'].append(f"erro_leitura:{str(e)[:60]}")
            stats['corrupted'].append(filename)
            records.append(record)
            continue

        stats, record = verificar_qualidade(img_path, config, stats, record)

        # Status final do arquivo
        if record['issues']:
            record['status'] = 'WARNING'
        else:
            stats['ok'] += 1

        records.append(record)

        # Log de progresso a cada 5%
        if (i + 1) % intervalo_log == 0:
            pct     = (i + 1) / qtd_imagens_encontradas * 100
            elapsed = time.time() - inicio
            eta     = (elapsed / (i + 1)) * (qtd_imagens_encontradas - i - 1)
            print(f"  Progresso: {i+1:,}/{qtd_imagens_encontradas:,} ({pct:.0f}%) | "
                  f"Tempo: {elapsed:.0f}s | ETA: {eta:.0f}s")

    elapsed_total = time.time() - inicio
    print(f"\n  Auditoria concluída em {elapsed_total:.1f}s")
    print(f"  Taxa: {qtd_imagens_encontradas / elapsed_total:.0f} imagens/segundo")

    return {
        'stats':   stats,
        'records': records,
        'elapsed': elapsed_total,
    }       


# auditar_corpus_fisico(CONFIG['image_dir'], CONFIG)


# # Cruzamento de cobertura com o CSVs

# In[51]:


def cruzar_cobertura_csv(resultado: dict, config: dict) -> dict:
    """
    Após a auditoria física, cruza os filenames auditados com os 5 CSVs
    para verificar cobertura tabular de cada imagem.

    Retorna dicionário com contagem de cobertura por dataset.
    """
    print_secao("CRUZAMENTO DE COBERTURA COM OS DATASETS TABULARES")

    # Conjunto de filenames físicos auditados
    filenames_fisicos = set(r['filename'] for r in resultado['records'])

    cobertura = {}
    datasets_dir = config['datasets_dir']

    for nome, arquivo in config['csv_files'].items():
        path = os.path.join(datasets_dir, arquivo)
        if not os.path.exists(path):
            print(f"  [{nome}] CSV não encontrado: {path}")
            cobertura[nome] = {'total': 0, 'com_imagem': 0, 'sem_imagem': 0}
            continue

        df = pd.read_csv(path)
        filenames_csv = set(
            normalizar_nome_arquivo(f) for f in df[config['filename_col']]
        )
        com_imagem  = len(filenames_csv & filenames_fisicos)
        sem_imagem  = len(filenames_csv - filenames_fisicos)
        cobertura[nome] = {
            'total':      len(df),
            'com_imagem': com_imagem,
            'sem_imagem': sem_imagem,
        }
        print(f"  [{nome:<15}] Total: {len(df):>5,} | "
              f"Com imagem: {com_imagem:>5,} | "
              f"Sem imagem: {sem_imagem:>3,}")

    # Imagens sem registro tabular em nenhum dataset
    todos_filenames_csv = set()
    for nome, arquivo in config['csv_files'].items():
        path = os.path.join(datasets_dir, arquivo)
        if os.path.exists(path):
            df = pd.read_csv(path)
            todos_filenames_csv |= set(
                normalizar_nome_arquivo(f) for f in df[config['filename_col']]
            )

    sem_registro = filenames_fisicos - todos_filenames_csv
    print(f"\n  Imagens sem registro tabular em nenhum dataset: {len(sem_registro):,}")
    if sem_registro:
        print("  Exemplos:", sorted(sem_registro)[:5])

    return cobertura


# # Relatório e exportação

# In[ ]:


def imprimir_relatorio(resultado: dict, cobertura: dict, config: dict):
    """Imprime relatório consolidado no console."""
    stats    = resultado['stats']
    qtd_imagens_encontradas = stats['qtd_imagens_encontradas']

    print_secao("RELATÓRIO DE INTEGRIDADE")

    print(f"\n  Diretório auditado : {config['image_dir']}")
    print(f"  Data/hora          : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

    print(f"\n  RESUMO GERAL")
    print(f"  {'Arquivos PNG no diretório':<40}: {qtd_imagens_encontradas:>8,}")
    delta = qtd_imagens_encontradas - stats['expected_total']
    print(f"  {'Contagem esperada':<40}: {stats['expected_total']:>8,}  "
          f"[{'OK' if delta == 0 else f'DIVERGÊNCIA {delta:+d}'}]")
    print(f"  {'Imagens verificadas individualmente':<40}: {stats['verificadas']:>8,}")
    print(f"  {'Imagens OK (sem problemas)':<40}: {stats['ok']:>8,}  "
          f"({stats['ok']/qtd_imagens_encontradas*100:.2f}%)")

    print(f"\n  PROBLEMAS DETECTADOS")
    categorias = [
        ('Arquivos corrompidos',      'corrupted'),
        ('Resolução incorreta',       'wrong_resolution'),
        ('Modo de cor incorreto',     'wrong_mode'),
        ('Brilho insuficiente',       'low_brightness'),
        ('Contraste insuficiente',    'low_contrast'),
        ('Arquivos muito grandes',    'oversized'),
    ]
    total_prob = 0
    for label, key in categorias:
        n = len(stats[key])
        total_prob += n
        flag = '  [ATENÇÃO]' if n > 0 else ''
        print(f"  {label:<40}: {n:>6,}{flag}")

    print(f"\n  {'TOTAL DE IMAGENS COM PROBLEMAS':<40}: {total_prob:>6,}  "
          f"({total_prob/qtd_imagens_encontradas*100:.2f}%)")

    integridade = stats['ok'] / qtd_imagens_encontradas * 100
    print(f"\n  TAXA DE INTEGRIDADE GERAL: {integridade:.2f}%")
    if integridade == 100.0:
        print("  STATUS: CORPUS ÍNTEGRO — Pronto para pipeline de Deep Learning")
    elif integridade >= 99.0:
        print("  STATUS: CORPUS ACEITÁVEL — Verificar imagens com problemas")
    else:
        print("  STATUS: ATENÇÃO — Taxa de problemas acima do threshold aceitável")


def exportar_csv(resultado: dict, config: dict):
    """Exporta relatório detalhado em CSV, um registro por arquivo."""
    os.makedirs(config['output_dir'], exist_ok=True)
    output_path = os.path.join(config['output_dir'], 'integrity_report.csv')

    fieldnames = [
        'filename', 'status', 'issues',
        'width', 'height', 'mode', 'file_size_bytes',
        'brightness_mean', 'contrast',
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for rec in resultado['records']:
            row = dict(rec)
            row['issues'] = '; '.join(row['issues']) if row['issues'] else 'nenhum'
            writer.writerow(row)

    print(f"\n  Relatório CSV exportado : {output_path}")
    print(f"  Total de registros      : {len(resultado['records']):,}")


def exportar_summary(resultado: dict, cobertura: dict, config: dict):
    """Exporta resumo textual completo."""
    stats     = resultado['stats']
    qtd_imagens_encontradas = stats['qtd_imagens_encontradas']
    output_path = os.path.join(config['output_dir'], 'integrity_summary.txt')
    os.makedirs(config['output_dir'], exist_ok=True)

    linhas = [
        "=" * 70,
        "RELATÓRIO DE INTEGRIDADE DAS IMAGENS ECG",
        "=" * 70,
        f"Projeto     : {config['project']}",
        f"Autor       : {config['author']}",
        f"Instituição : {config['institution']}",
        f"Data/Hora   : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
        f"Script      : script_07_verificar_integridade_imagens.ipynb "
        "",
        "AUDITORIA FÍSICA",
        "-" * 40,
        f"Arquivos PNG no diretório  : {qtd_imagens_encontradas:,}",
        f"Contagem esperada          : {stats['expected_total']:,}",
        f"Delta (físico - esperado)  : {qtd_imagens_encontradas - stats['expected_total']:+d}",
        f"Imagens verificadas        : {stats['verificadas']:,}",
        f"Imagens OK                 : {stats['ok']:,} ({stats['ok']/qtd_imagens_encontradas*100:.2f}%)",
        f"Corrompidas                : {len(stats['corrupted']):,}",
        f"Resolução incorreta        : {len(stats['wrong_resolution']):,}",
        f"Modo de cor incorreto      : {len(stats['wrong_mode']):,}",
        f"Brilho insuficiente        : {len(stats['low_brightness']):,}",
        f"Contraste insuficiente     : {len(stats['low_contrast']):,}",
        f"Arquivos muito grandes     : {len(stats['oversized']):,}",
        "",
        f"TAXA DE INTEGRIDADE GERAL  : {stats['ok']/qtd_imagens_encontradas*100:.2f}%",
        f"TEMPO DE EXECUÇÃO          : {resultado['elapsed']:.1f}s",
        "",
        "COBERTURA TABULAR POR DATASET",
        "-" * 40,
    ]

    for nome, cob in cobertura.items():
        pct = cob['com_imagem'] / cob['total'] * 100 if cob['total'] > 0 else 0
        linhas.append(
            f"{nome:<20}: {cob['total']:>5,} registros | "
            f"Com imagem: {cob['com_imagem']:>5,} ({pct:.1f}%) | "
            f"Sem imagem: {cob['sem_imagem']:>3,}"
        )

    linhas.append("=" * 70)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(linhas))

    print(f"  Summary exportado       : {output_path}")


# # Gráficos

# In[53]:


def gerar_visualizacoes(resultado: dict, config: dict):
    """Gera painel de 6 gráficos de qualidade do corpus."""
    os.makedirs(config['plots_dir'], exist_ok=True)

    records   = resultado['records']
    stats     = resultado['stats']
    qtd_imagens_encontradas = stats['qtd_imagens_encontradas']

    df_rec = pd.DataFrame(records)
    df_ok  = df_rec[
        df_rec['brightness_mean'].notna() & (df_rec['brightness_mean'] >= 0)
    ]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f'Análise de Qualidade das Imagens ECG\n'
        f'Corpus físico: {qtd_imagens_encontradas:,} arquivos PNG | '
        f'Integridade: {stats["ok"]/qtd_imagens_encontradas*100:.2f}%',
        fontsize=13, fontweight='bold', y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Plot 1: Status geral
    ax1 = fig.add_subplot(gs[0, 0])
    n_warn = sum(1 for r in records if r['status'] == 'WARNING')
    dados_pizza = [
        ('OK',        stats['ok'],              '#2ecc71'),
        ('WARNING',   n_warn,                   '#f39c12'),
        ('CORRUPTED', len(stats['corrupted']),  '#8e44ad'),
    ]
    labels_p = [f"{l}\n({v:,})" for l, v, _ in dados_pizza if v > 0]
    sizes_p  = [v for _, v, _ in dados_pizza if v > 0]
    colors_p = [c for _, v, c in dados_pizza if v > 0]
    if sizes_p:
        ax1.pie(sizes_p, labels=labels_p, colors=colors_p,
                autopct='%1.1f%%', startangle=90)
    ax1.set_title('Status das Imagens', fontweight='bold')

    # Plot 2: Distribuição de brilho
    ax2 = fig.add_subplot(gs[0, 1])
    if len(df_ok) > 0:
        ax2.hist(df_ok['brightness_mean'].dropna(), bins=50,
                 color='#3498db', edgecolor='white', linewidth=0.5)
        ax2.axvline(config['min_brightness'], color='red', linestyle='--',
                    linewidth=1.5, label=f"Mín={config['min_brightness']}")
        ax2.axvline(config['max_brightness'], color='orange', linestyle='--',
                    linewidth=1.5, label=f"Máx={config['max_brightness']}")
        ax2.set_xlabel('Brilho Médio (0-255)')
        ax2.set_ylabel('Frequência')
        ax2.set_title('Distribuição de Brilho', fontweight='bold')
        ax2.legend(fontsize=8)

    # Plot 3: Distribuição de contraste
    ax3 = fig.add_subplot(gs[0, 2])
    if len(df_ok) > 0:
        ax3.hist(df_ok['contrast'].dropna(), bins=50,
                 color='#9b59b6', edgecolor='white', linewidth=0.5)
        ax3.axvline(config['min_contrast'], color='red', linestyle='--',
                    linewidth=1.5, label=f"Mín={config['min_contrast']}")
        ax3.set_xlabel('Contraste (Desvio Padrão)')
        ax3.set_ylabel('Frequência')
        ax3.set_title('Distribuição de Contraste', fontweight='bold')
        ax3.legend(fontsize=8)

    # Plot 4: Scatter brilho x contraste
    ax4 = fig.add_subplot(gs[1, 0])
    if len(df_ok) > 0:
        amostra = df_ok.sample(min(1000, len(df_ok)), random_state=42)
        ax4.scatter(amostra['brightness_mean'], amostra['contrast'],
                    alpha=0.3, s=5, color='#1abc9c')
        ax4.set_xlabel('Brilho Médio')
        ax4.set_ylabel('Contraste')
        ax4.set_title('Brilho × Contraste', fontweight='bold')

    # Plot 5: Distribuição de tamanho de arquivo
    ax5 = fig.add_subplot(gs[1, 1])
    df_sz = df_rec[df_rec['file_size_bytes'].notna()]
    if len(df_sz) > 0:
        sizes_mb = df_sz['file_size_bytes'] / (1024 * 1024)
        ax5.hist(sizes_mb, bins=50, color='#e67e22',
                 edgecolor='white', linewidth=0.5)
        ax5.set_xlabel('Tamanho (MB)')
        ax5.set_ylabel('Frequência')
        ax5.set_title('Distribuição de Tamanho', fontweight='bold')

    # Plot 6: Resumo numérico
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    integridade = stats['ok'] / qtd_imagens_encontradas * 100
    resumo = (
        f"RESUMO GERAL\n"
        f"{'='*28}\n"
        f"Total físico     : {qtd_imagens_encontradas:,}\n"
        f"Esperado         : {stats['expected_total']:,}\n"
        f"Integridade      : {integridade:.2f}%\n"
        f"OK               : {stats['ok']:,}\n"
        f"Corrompidas      : {len(stats['corrupted']):,}\n"
        f"Res. incorreta   : {len(stats['wrong_resolution']):,}\n"
        f"Modo incorreto   : {len(stats['wrong_mode']):,}\n"
        f"Baixo brilho     : {len(stats['low_brightness']):,}\n"
        f"Baixo contraste  : {len(stats['low_contrast']):,}\n"
    )
    ax6.text(0.05, 0.95, resumo, transform=ax6.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))

    output_path = os.path.join(config['plots_dir'], 'image_quality_stats.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Visualização exportada  : {output_path}")


# In[54]:


def main():
    print_separador()
    print("  SCRIPT 07: VERIFICAÇÃO DE INTEGRIDADE DAS IMAGENS ECG")
    print(f"  {CONFIG['institution']}")
    print(f"  {CONFIG['project']}")
    print(f"  Autor  : {CONFIG['author']}")
    print(f"  Data   : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print_separador()

    if not os.path.exists(CONFIG['image_dir']):
        print(f"\n  ERRO: Diretório de imagens não encontrado: {CONFIG['image_dir']}")
        return

    # ------------------------------------------------------------------
    # 1. Auditoria física — itera sobre os 3.900 arquivos do diretório
    # ------------------------------------------------------------------
    resultado = auditar_corpus_fisico(CONFIG['image_dir'], CONFIG)

    # ------------------------------------------------------------------
    # 2. Cruzamento de cobertura com os 5 CSVs
    # ------------------------------------------------------------------
    cobertura = cruzar_cobertura_csv(resultado, CONFIG)

    # ------------------------------------------------------------------
    # 3. Relatório consolidado
    # ------------------------------------------------------------------
    imprimir_relatorio(resultado, cobertura, CONFIG)

    # ------------------------------------------------------------------
    # 4. Exportar artefatos
    # ------------------------------------------------------------------
    print_secao("EXPORTANDO ARTEFATOS")
    exportar_csv(resultado, CONFIG)
    exportar_summary(resultado, cobertura, CONFIG)
    gerar_visualizacoes(resultado, CONFIG)

    print_separador()
    print("  SCRIPT 07 CONCLUÍDO COM SUCESSO")
    print_separador()


if __name__ == '__main__':
    main()

