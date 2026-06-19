"""
# Script 06 - Modelos tabulares

## Descrição
Este scritp apresenta os 6 modelos tabulares, sendo eles: Random Forest, XGBoost, Gradient Boosting, Support Vector Machines (SVM), Multilayer Perceptron (MLP) e Logistic Regression

## Objetivo 
- Treinar os 6 modelos com os dados tabulares presentes no dataset SILVER.
- Gerar métricas de avaliação dos 6 modelos para análises posteriores.

## 

"""

import os
import warnings
from datetime import datetime

import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)

from config import build_config_06


warnings.filterwarnings("ignore")

optuna.logging.set_verbosity(optuna.logging.WARNING)


def print_separador(char='=', largura=80):
    print(char * largura)


def log_message(message):
    """
    Função para imprimir logs

    Args: 
        message: str
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] - {message}")


def clip_outliers(df, ranges_dict, output_dir):
    """
    Detectar e cortar ("clipar") os outliers
    Args:
        df: datagrame
        ranges_dict: dicionários dos cortes
        output_dir: pasta para salvar o relatório
    Returns: 
        df_clipped: dataframe sem os outliers
        outliers_stats: estatísticas dos outliers que foram corrigidos
    """
    log_message("Analisando e tratando outliers com ranges ajustados (v1.2)...")

    df_clipped = df.copy()
    outliers_stats = {}
    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append("RELATORIO DE TRATAMENTO DE OUTLIERS (v1.2)")
    report_lines.append("=" * 80)
    report_lines.append(
        "\nVersao: Script v1.2 - Ranges Ajustados para Patologias Reais")
    report_lines.append(f"Dataset: {len(df)} registros")
    report_lines.append(
        f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    total_outliers = 0

    for col, (min_val, max_val) in ranges_dict.items():
        below_min = (df[col] < min_val).sum()
        above_max = (df[col] > max_val).sum()
        n_outliers = below_min + above_max

        if n_outliers > 0:
            original_min = df[col].min()
            original_max = df[col].max()

            df_clipped[col] = df_clipped[col].clip(
                lower=min_val, upper=max_val)

            outliers_stats[col] = {
                "n_outliers": n_outliers,
                "percent": n_outliers / len(df) * 100,
                "below_min": below_min,
                "above_max": above_max,
                "original_min": original_min,
                "original_max": original_max,
                "range_min": min_val,
                "range_max": max_val
            }

            total_outliers += n_outliers

            report_lines.append(f"\n{'-' * 80}")
            report_lines.append(f"Parametro: {col}")
            report_lines.append(f"  Range ajustado: [{min_val}, {max_val}]")
            report_lines.append(
                f"  Outliers encontrados: {n_outliers} ({n_outliers/len(df)*100:.2f}%)"
            )
            if below_min > 0:
                report_lines.append(
                    f"    - Abaixo do minimo: {below_min} "
                    f"(valor min original: {original_min:.2f})"
                )
            if above_max > 0:
                report_lines.append(
                    f"    - Acima do maximo: {above_max} "
                    f"(valor max original: {original_max:.2f})"
                )
            report_lines.append(
                f"  Acao: Valores clipados para range [{min_val}, {max_val}]"
            )

    report_lines.append(f"\n{'=' * 80}")
    report_lines.append("RESUMO")
    report_lines.append(f"{'=' * 80}")
    report_lines.append(
        f"Total de parametros com outliers: {len(outliers_stats)}/14")
    report_lines.append(f"Total de valores clipados: {total_outliers}")
    report_lines.append(
        f"Percentual de valores afetados: {total_outliers/(len(df)*14)*100:.2f}%"
    )
    report_lines.append(f"\nTodos os registros foram mantidos (N={len(df)})")
    report_lines.append(
        "Metodo aplicado: CLIPPING (valores limitados aos ranges ajustados)")
    report_lines.append(
        "\nRanges incluem valores patologicos mas biologicamente possiveis")

    output_file = output_dir + "/outliers_stats.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log_message(
        f"Relatorio de outliers salvo: [{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] - {output_file}")

    if outliers_stats:
        log_message(
            f"Outliers tratados: {len(outliers_stats)} parametros, "
            f"{total_outliers} valores clipados"
        )
        for col, stats in outliers_stats.items():
            log_message(
                f"  {col}: {stats['n_outliers']} outliers ({stats['percent']:.2f}%)"
            )
    else:
        log_message(
            "Nenhum outlier encontrado! Dataset dentro dos ranges ajustados.")

    return df_clipped, outliers_stats


def validate_dataset(df, label_col, param_cols):
    """
    Validar o dataset, com a quantidade de colunas esperadas, detectar a presença de NaN

    Args:
        df: dataframe
    Returns:
        bool: retorna True se o datrafame está ok 
    """
    log_message("Validando estrutura do dataset...")

    assert len(
        df.columns) == 16, f"Esperado 16 colunas, encontrado {len(df.columns)}"
    assert label_col in df.columns, "Coluna 'classificacao' ausente"
    assert df[label_col].dtype == "int64", (
        f"Classificacao deve ser int64, encontrado {df[label_col].dtype}"
    )

    unique_values = set(df[label_col].unique())
    assert unique_values == {0, 1}, (
        f"Classificacao deve conter apenas {{0, 1}}, encontrado {unique_values}"
    )

    assert df[label_col].isnull().sum(
    ) == 0, "Encontrados NaN em classificacao"

    nan_counts = df[param_cols].isnull().sum()
    if nan_counts.sum() > 0:
        log_message("AVISO: NaN encontrados nas features:")
        for col, count in nan_counts[nan_counts > 0].items():
            log_message(f"  {col}: {count} NaN ({count/len(df)*100:.2f}%)")
        raise ValueError(
            "Dataset contem NaN nas features. Verificar etapa de imputacao.")

    log_message("Estrutura validada com sucesso!")
    return True


def load_and_prepare_data(config: dict):
    """
    Carregar e preparar os dados para as etapas subsequentes

    Returns:
        X_train_scaled: dataframe X para treino com o uso do fit_transform (aprender com os parâmetros e transforma eles)
        X_val_scaled: dataframe X para treino com o uso do transform (apenas transforma)
        X_test_scaled: dataframe X para treino com o uso do transform
        y_train: dataframe y para treino
        y_val: dataframe y para validação
        y_test: dataframe y para teste
        FEATURE_COLS: colunas de features
        scaler: scaler usado
        outliers_stats: estatísticas dos otliers
    )
    """
    log_message(f"Carregando o dataset {config['datasets']['SILVER']['file']}")

    df = pd.read_csv(config['datasets']['SILVER']['file'])
    validate_dataset(
        df, label_col=config['label_col'], param_cols=config['param_cols'])

    df_clean, outliers_stats = clip_outliers(
        df, config['physiological_ranges'], config['output_dir'])

    X = df_clean[config['param_cols']].copy()
    y = df_clean[config['label_col']].copy()

    # Conjunto de testes
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=config['test_ratio'],
        random_state=config['random_seed'],
        stratify=y
    )

    # Conjunto de treino e validação
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=config['val_ratio'],
        random_state=config['random_seed'],
        stratify=y_temp
    )

    log_message(
        f"Train set: {len(X_train)} registros ({len(X_train)/len(X)*100:.1f}%)")
    log_message(
        f"Val set:   {len(X_val)} registros ({len(X_val)/len(X)*100:.1f}%)")
    log_message(
        f"Test set:  {len(X_test)} registros ({len(X_test)/len(X)*100:.1f}%)")

    log_message("Aplicação do scaler")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    log_message(
        f"Media apos escala (train): {X_train_scaled.mean():.4f} (esperado ~0)")
    log_message(
        f"Desvio padrao apos escala (train): {X_train_scaled.std():.4f} (esperado ~1)")

    return (
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test,
        config['param_cols'], scaler, outliers_stats
    )


def make_pipeline_objective(model_fn, metric, cv_folds, random_seed):
    """Envolve qualquer modelo num Pipeline com StandardScaler."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                         random_state=random_seed)

    def objective(trial, X, y):
        model = model_fn(trial, random_seed)
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        return cross_val_score(pipe, X, y, cv=cv, scoring=metric).mean()
    return objective


# objetivos de cada um dos modelos
def make_lr(trial, random_seed):
    return LogisticRegression(
        C=trial.suggest_float("C", 1e-3, 1e2, log=True),
        penalty=trial.suggest_categorical("penalty", ["l1", "l2"]),
        solver="liblinear",
        max_iter=1000,
        random_state=random_seed,
    )


def make_svm(trial, random_seed):
    return SVC(
        C=trial.suggest_float("C", 1e-2, 1e3, log=True),
        kernel=trial.suggest_categorical("kernel", ["rbf", "linear"]),
        gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),
        probability=True,
        random_state=random_seed,
    )


def make_mlp(trial, random_seed):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = tuple(trial.suggest_int(
        f"units_l{i}", 32, 128) for i in range(n_layers))
    return MLPClassifier(
        hidden_layer_sizes=layers,
        activation=trial.suggest_categorical("activation", ["relu", "tanh"]),
        learning_rate_init=trial.suggest_float(
            "learning_rate_init", 1e-4, 1e-2, log=True),
        alpha=trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        solver="sgd",
        learning_rate="adaptive",
        nesterovs_momentum=True,
        early_stopping=True,
        max_iter=1000,
        random_state=random_seed,
    )


def make_xgb(trial, random_seed):
    return xgb.XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        max_depth=trial.suggest_int("max_depth", 3, 6),
        learning_rate=trial.suggest_float("learning_rate", 0.05, 0.3),
        subsample=trial.suggest_float("subsample", 0.6, 0.9),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 0.9),
        min_child_weight=trial.suggest_int("min_child_weight", 5, 20),
        reg_alpha=trial.suggest_float("reg_alpha", 0.1, 10.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0),
        eval_metric="logloss", random_state=random_seed,
        n_jobs=-1,
    )


def make_gb(trial, random_seed):
    return GradientBoostingClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 400),
        max_depth=trial.suggest_int("max_depth", 3, 6),
        learning_rate=trial.suggest_float("learning_rate", 0.05, 0.3),
        subsample=trial.suggest_float("subsample", 0.6, 0.9),
        min_samples_split=trial.suggest_int("min_samples_split", 10, 30),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 20),
        random_state=random_seed,
    )


def make_rf(trial, random_seed):
    return RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 400),
        max_depth=trial.suggest_int("max_depth", 5, 20),
        min_samples_split=trial.suggest_int("min_samples_split", 10, 30),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 20),
        max_features=trial.suggest_categorical(
            "max_features", ["sqrt", "log2"]),
        random_state=random_seed,
        n_jobs=-1,
    )


def tune_all_models(X_train, y_train, config):
    """
    Ajuste (otimização) dos hiperparâmetros

    Args:
        X_train: dataframe X para treinos
        y_train: dataframe y para treinos
        config: dicionário

    Returns:
        best_params: melhores parâmetros de cada modelo
    """
    log_message("OTIMIZAÇÃO DE HIPERPARÂMETROS (Optuna)")
    best_params = {}

    objetives = {
        "Logistic Regression": make_pipeline_objective(make_lr, config['metric'], config['cv_folds'], config['random_seed']),
        "Random Forest": make_pipeline_objective(make_rf, config['metric'], config['cv_folds'], config['random_seed']),
        "XGBoost": make_pipeline_objective(make_xgb, config['metric'], config['cv_folds'], config['random_seed']),
        "Gradient Boosting": make_pipeline_objective(make_gb, config['metric'], config['cv_folds'], config['random_seed']),
        "SVM": make_pipeline_objective(make_svm, config['metric'], config['cv_folds'], config['random_seed']),
        "MLP": make_pipeline_objective(make_mlp, config['metric'], config['cv_folds'], config['random_seed']),
    }

    for name, objective in objetives.items():
        log_message(f"  Otimizando {name}...")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=config['random_seed'])
        )
        study.optimize(
            lambda trial: objective(
                trial, X_train, y_train),
            n_trials=config['n_trials'],
            show_progress_bar=True
        )

        best_params[name] = study.best_params
        log_message(
            f"  ✔ {name}: {config['metric']} = {study.best_value:.4f} | params: {study.best_params}")

    return best_params


# BUILD — reconstrói modelos com params ótimos
def build_optimized_models(best_params, y_train, config):
    """
    Reconstrução dos modelos com os mehores parâmetros

    Args: 
        best_params: melhores parâmetros
        y_train: dataframe y para treinos

    Return:
        Object: modelos
    """
    p = best_params

    layers = tuple(
        p["MLP"][f"units_l{i}"]
        for i in range(p["MLP"]["n_layers"])
    )

    return {
        "Logistic Regression": LogisticRegression(
            **{k: v for k, v in p["Logistic Regression"].items()},
            class_weight="balanced",
            solver="liblinear",
            max_iter=1000,
            random_state=config['random_seed'],
        ),
        "Random Forest": RandomForestClassifier(
            **{k: v for k, v in p["Random Forest"].items()},
            class_weight="balanced",
            random_state=config['random_seed'],
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            **{k: v for k, v in p["XGBoost"].items()},
            scale_pos_weight=sum(y_train == 0) / sum(y_train == 1),
            eval_metric="logloss",
            random_state=config['random_seed'],
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            **{k: v for k, v in p["Gradient Boosting"].items()},
            random_state=config['random_seed'],
        ),
        "SVM": SVC(
            **{k: v for k, v in p["SVM"].items()},
            class_weight="balanced",
            probability=True,
            random_state=config['random_seed'],
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=layers,
            activation=p["MLP"]["activation"],
            learning_rate_init=p["MLP"]["learning_rate_init"],
            alpha=p["MLP"]["alpha"],
            solver="sgd",
            learning_rate="adaptive",
            nesterovs_momentum=True,
            early_stopping=True,
            max_iter=1000,
            random_state=config['random_seed'],
        ),
    }


def evaluate_model(model, X_test, y_test):
    """
    Avaliação do modelo

    Args: 
        model: modelo
        X_test: dataframe X para teste
        y_test: dataframe y para teste

    Returns:
        metrics: métricas dos modelos
        y_pred: predições 
        y_prob: probabilidade
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_prob),
    }
    return metrics, y_pred, y_prob


def save_all_results(results, trained_models, best_params, config):
    """
    Salva modelos, métricas e hiperparâmetros em results/.

    Estrutura gerada:
        output_dir/
        ├── models/          → .pkl de cada modelo
        ├── metrics.csv      → tabela comparativa
        └── hyperparams.json → melhores params do Optuna
    """
    base = config['output_dir']
    models_dir = os.path.join(base, "models")

    os.makedirs(models_dir, exist_ok=True)

    # Salva os modelos (.pkl)
    for name, data in trained_models.items():
        filename = name.lower().replace(" ", "_") + ".pkl"
        filepath = os.path.join(models_dir, filename)
        joblib.dump(data["model"], filepath)
        log_message(f"  Modelo salvo: {filepath}")

    # Salva as métricas (.csv)
    df_metrics = (
        pd.DataFrame(results)
        .T
        .rename_axis("model")
        .reset_index()
        .sort_values("auc_roc", ascending=False)
    )
    metrics_path = os.path.join(base, "metrics.csv")
    df_metrics.to_csv(metrics_path, index=False, float_format="%.4f")
    log_message(f"  Métricas salvas: {metrics_path}")

    # Salva os hiperparâmetros (.json)
    params_path = os.path.join(base, "hyperparams.json")
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    log_message(f"  Hiperparâmetros salvos: {params_path}")

    log_message(f"✔ Resultados salvos em: {base}/")
    return base


def train_and_evaluate_all(X_train, y_train, X_val, y_val, X_test, y_test, config):
    """
    Treinar e evaliar todos os modelos

    Args: 
        X_train: dataframe X para treino
        y_train: dataframe y para treino
        X_val: dataframe X para validação
        y_val: dataframe y para validação
        X_test: dataframe X para teste
        y_test: dataframe y para teste

    Return:
        results: resultados
        trained_models: modelos treinados
        best_params: melhores parâmetros
    """
    best_params = tune_all_models(X_train, y_train, config)

    models = build_optimized_models(best_params, y_train, config)

    log_message("TREINAMENTO DOS MODELOS")
    results = {}
    trained_models = {}

    for name, model in models.items():
        log_message(f"Treinando o modelo {name}")

        # XGBoost usa X_val para early stopping
        if name == "XGBoost":
            model.fit(X_train, y_train, eval_set=[
                      (X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)

        metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test)

        log_message(f"  Accuracy:  {metrics['accuracy']:.4f}")
        log_message(f"  Precision: {metrics['precision']:.4f}")
        log_message(f"  Recall:    {metrics['recall']:.4f}")
        log_message(f"  F1-Score:  {metrics['f1']:.4f}")
        log_message(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")

        results[name] = metrics
        trained_models[name] = {"model": model,
                                "y_pred": y_pred, "y_proba": y_proba}

    # 3) Ranking final
    log_message("RANKING FINAL (AUC-ROC)")
    ranking = sorted(
        results.items(), key=lambda x: x[1]["auc_roc"], reverse=True)
    for i, (name, m) in enumerate(ranking, 1):
        log_message(
            f"  {i}. {name}: AUC-ROC={m['auc_roc']:.4f}  F1={m['f1']:.4f}")

    save_all_results(results, trained_models, best_params, config)

    return results, trained_models, best_params


# Treinar e avaliar todos os modelos


# ─────────────────────────────────────────────
# CARREGA hiperparâmetros salvos
# ─────────────────────────────────────────────

def load_best_params(results_dir):
    """
    Carregar os melhores hiperparâmetros

    Args: 
        results_dir: diretórios dos resultados
    Returns: 
        best_params: melhores hiperparâmetros
    """
    params_path = os.path.join(results_dir, "hyperparams.json")
    with open(params_path, "r") as f:
        best_params = json.load(f)
    log_message(f"Hiperparâmetros carregados de: {params_path}")
    return best_params


def train_with_best_params(best_params, X_train, y_train, X_val, y_val, config):
    """
    Treinar com os melhores prâmetros

    Args: 
        best_params: melhores parâmetros
        X_train: dataframe X para treino
        y_train: dataframe y para treino
        X_val: dataframe X para validação
        y_val: dataframe y para validação

    Returns:
        trained_models: modelos treinados
    """
    log_message("TREINAMENTO COM MELHORES PARÂMETROS")

    models = build_optimized_models(best_params, y_train, config)
    trained_models = {}

    for name, model in models.items():
        log_message(f"  Treinando {name}...")

        if name == "XGBoost":
            model.fit(X_train, y_train, eval_set=[
                      (X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)

        trained_models[name] = model

    return trained_models


def evaluate_all(trained_models, X_test, y_test):
    """
    Avaliar os modelos 

    Args: 
        trained_models: modelos treinados
        X_test: dataframe X para teste
        y_test: dataframe y para teste

    Returns: 
        results: resultados dos treinos, composto pelas métricas (accuracy, precision, recall, f1, auc_roc), y_pred e y_prob
    """
    log_message("AVALIAÇÃO DOS MODELOS")

    results = {}

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall":    recall_score(y_test, y_pred),
            "f1":        f1_score(y_test, y_pred),
            "auc_roc":   roc_auc_score(y_test, y_prob),
        }

        results[name] = {
            "metrics": metrics,
            "y_pred":  y_pred,
            "y_prob": y_prob
        }

        log_message(f"  {name}")
        for metric, value in metrics.items():
            log_message(f"    {metric}: {value:.4f}")

    return results


def save_trained_models(trained_models, results, config):
    """
    Salvar os modelos treinados e suas métricas

    Args: 
        trained_models: modelos treinados
        results: resultados

    Returns: 
        base: diretório
    """
    base = config['output_dir']
    models_dir = os.path.join(base, "models")
    os.makedirs(models_dir, exist_ok=True)

    # .pkl de cada modelo
    for name, model in trained_models.items():
        filename = name.lower().replace(" ", "_") + ".pkl"
        filepath = os.path.join(models_dir, filename)
        joblib.dump(model, filepath)
        log_message(f"  Modelo salvo: {filepath}")

    # métricas em CSV
    df_metrics = pd.DataFrame({
        name: data["metrics"] for name, data in results.items()
    }).T

    df_metrics = df_metrics.sort_values("auc_roc", ascending=False)

    metrics_path = os.path.join(base, "metrics.csv")
    df_metrics.to_csv(metrics_path, index=True, float_format="%.4f")
    log_message(f"  Métricas salvas: {metrics_path}")

    log_message(f"✔ Tudo salvo em: {base}/")
    return base


def retrain_and_save(results_dir, X_train, y_train, X_val, y_val, X_test, y_test, config):
    """
    Fluxo completo:
        1. Carrega hiperparâmetros do JSON salvo anteriormente
        2. Treina os modelos com esses parâmetros
        3. Avalia no conjunto de teste
        4. Salva os .pkl e o metrics.csv

    Args:
        results_dir : pasta gerada pelo save_all_results() anterior, ex: "results_20250416_143022"
    """
    best_params = load_best_params(results_dir)
    trained_models = train_with_best_params(
        best_params, X_train, y_train, X_val, y_val, config)
    results = evaluate_all(trained_models, X_test, y_test)
    output_dir = save_trained_models(trained_models, results, config)

    return trained_models, results, output_dir


def plot_comparison(results, output_path):
    """
    Gera um gráfico de barras comparando o desempenho de múltiplos modelos

    Args:
        results: resultados dos treinos, composto pelas métricas (accuracy, precision, recall, f1, auc_roc), y_pred e y_prob
        output_path: diretório para salvar os gráficos
    """
    log_message("Gerando grafico de comparacao de modelos...")
    df_results = pd.DataFrame({
        name: data["metrics"] for name, data in results.items()
    }).T.sort_values("auc_roc", ascending=False)

    df_results = df_results[["accuracy",
                             "precision", "recall", "f1", "auc_roc"]]

    fig, ax = plt.subplots(figsize=(12, 6))
    df_results.plot(kind="bar", ax=ax, width=0.8)
    plt.title("Comparacao de Performance - Modelos Baseline",
              fontsize=14, fontweight="bold")
    plt.xlabel("Modelo", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(loc="lower right", fontsize=10)
    plt.ylim([0.7, 1.0])
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    log_message(f"Grafico salvo: {output_path}")


def plot_feature_importance(model, feature_names, model_name, output_path):
    """
    Gera um gráfico de importância das features para um modelo treinado

    Args: 
        model: modelos
        feature_names: nomes das features
        model_name: nome do modelo
        output_path: diretório para salvar os gráficos
    """
    log_message(f"Gerando feature importance para {model_name}...")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        log_message(
            f"  {model_name} nao suporta feature importance. Pulando...")
        return

    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(importances))
    ax.barh(y_pos, importances[indices], alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel("Importancia", fontsize=12)
    ax.set_title(
        f"Feature Importance - {model_name}", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    log_message(f"Grafico salvo: {output_path}")

    log_message(f"  Top 5 features ({model_name}):")
    for i in range(min(5, len(importances))):
        idx = indices[i]
        log_message(f"    {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")


def plot_confusion_matrix(y_test, y_pred, model_name, output_path):
    """
    Gera a matriz de confusão

    Args: 
        y_test: rótulos verdadeiros
        y_pred: rótulos preditos pelo modelo
        model_name: nome do modelo
        output_path: diretório para salvar os gráficos
    """
    log_message(f"Gerando confusion matrix para {model_name}...")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["NORMAL", "ANORMAL"],
        yticklabels=["NORMAL", "ANORMAL"],
        ax=ax, cbar_kws={"label": "Count"}
    )
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title(f"Confusion Matrix - {model_name}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    log_message(f"Grafico salvo: {output_path}")


def plot_roc_curves(trained_models, y_test, results, output_path):
    """
    Gera curvas ROC para múltiplos modelos em um único gráfico

    Args: 
        trained_models: modelos treinados
        y_test: rótulos verdadeiros
        results: resultados dos treinos, composto pelas métricas (accuracy, precision, recall, f1, auc_roc), y_pred e y_prob
        output_path: diretório para salvar os gráficos
    """
    log_message("Gerando curvas ROC comparativas...")

    fig, ax = plt.subplots(figsize=(10, 8))

    for name, model_data in trained_models.items():
        y_proba = results[name]["y_prob"]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves - Comparacao de Modelos",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    log_message(f"Grafico salvo: {output_path}")


def main():
    config = build_config_06()

    print_separador()
    print("  SCRIPT 06: TREINAMENTO DE MODELOS TABULARES")
    print(f"  {config['institution']}")
    print(f"  {config['project']}")
    print(f"  Autor  : {config['author']}")
    print(f"  Data   : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print_separador()

    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler, outliers_stats = load_and_prepare_data(
        config=config)

    results, trained_models, best_params = train_and_evaluate_all(
        X_train, y_train, X_val, y_val, X_test, y_test, config=config)

    trained_models, results, output_dir = retrain_and_save(
        results_dir=config['output_dir'],
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        config=config
    )

    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 10

    os.makedirs(config['plots_dir'], exist_ok=True)

    plot_comparison(results, os.path.join(
        config['plots_dir'], 'baseline_comparison.png'))

    for name, model in trained_models.items():
        plot_feature_importance(
            model,
            feature_names,
            name,
            os.path.join(config['plots_dir'], f"feature_importance_{name.lower().replace(' ', '_')}.png")
        )

    for name, data in results.items():
        plot_confusion_matrix(
            y_test,
            data["y_pred"],
            name,
            os.path.join(config['plots_dir'], f"confusion_matrix_{name.lower().replace(' ', '_')}.png")
        )

    plot_roc_curves(
        trained_models,
        y_test,
        results,
        os.path.join(config['plots_dir'], "roc_curves_comparison.png")
    )


if __name__ == '__main__':
    main()
