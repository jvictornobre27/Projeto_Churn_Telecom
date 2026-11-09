"""Motor do XGBoost: carga dos dados, treino com parada antecipada, previsão e busca com Optuna.

O notebook `06_xgboost.ipynb` só CHAMA este módulo. Segue as mesmas regras do MLP (`src/mlp.py`):
- Escolhas (hiperparâmetros, variante, etapa) usam SÓ o KS de validação (desempate: AUROC de validação).
  O teste só é carregado com `incluir_teste=True`, para registro.
- Busca de hiperparâmetros apenas no Fold 1; a configuração escolhida é retreinada nos Folds 1, 2 e 3
  com 3 sementes (`subsample`/`colsample_bytree` < 1 tornam o treino aleatório).
- Parada antecipada pela log-loss de validação (deviance, como no slide), paciência 20.

Variantes de entrada:
- `arvore`: categóricas como texto -> tipo `category` do pandas (suporte nativo do XGBoost).
  As categorias são aprendidas SÓ no treino do fold; categoria nova em val/teste vira ausente.
- `neural`: one-hot + Min-Max (a escala não muda nada para árvores; serve para comparar a codificação).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config as cfg
from .metrics import evaluate_sets, ks_score
from .preprocess import load_preprocessed

MODELO = "xgboost"
PACIENCIA = 20
MAX_ARVORES = 2000
OPTUNA_DIR = cfg.ROOT / "models" / "optuna"   # `models/` não vai para o Git


@dataclass
class XGBConfig:
    """Hiperparâmetros. O padrão (BASELINE) são os valores padrão da biblioteca."""
    learning_rate: float = 0.3
    max_depth: int = 6
    min_child_weight: float = 1.0     # equivalente ao min_samples_leaf do slide (peso mínimo por folha)
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    gamma: float = 0.0                # ganho mínimo para dividir um nó
    reg_lambda: float = 1.0           # regularização L2
    reg_alpha: float = 0.0            # regularização L1
    max_arvores: int = MAX_ARVORES    # teto; o número real sai da parada antecipada
    paciencia: int = PACIENCIA


BASELINE = XGBConfig()


def config_to_dict(c: XGBConfig) -> dict:
    return asdict(c)


_INTEIROS = ("max_depth", "max_arvores", "paciencia")


def config_from_dict(d: dict) -> XGBConfig:
    """Reconstrói a configuração (tabelas do pandas transformam inteiros em float; aqui voltam a int)."""
    kw = {k: d[k] for k in XGBConfig.__dataclass_fields__ if k in d}
    return XGBConfig(**{k: int(v) if k in _INTEIROS else float(v) for k, v in kw.items()})


def save_config(c: XGBConfig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_to_dict(c), indent=2), encoding="utf-8")
    return path


def load_config(path: Path) -> XGBConfig:
    return config_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# Dados
def _categorias_do_treino(X_tr: pd.DataFrame) -> dict:
    return {c: sorted(X_tr[c].astype(str).unique()) for c in cfg.CATEGORICAL_COLS if c in X_tr.columns}


def _como_category(X: pd.DataFrame, cats: dict) -> pd.DataFrame:
    X = X.copy()
    for c, valores in cats.items():
        X[c] = pd.Categorical(X[c].astype(str), categories=valores)
    return X


def load_xy(fold: int, variante: str, conjuntos=("treino", "validacao")) -> dict:
    """{conjunto: (X, y)}. Por padrão NÃO carrega o teste.

    Na variante `arvore`, as categorias vêm só do treino do fold (mesmo se o treino não for pedido).
    """
    data = {c: load_preprocessed(fold, c, variante) for c in conjuntos}
    if variante == "arvore":
        X_tr = data["treino"][0] if "treino" in data else load_preprocessed(fold, "treino", variante)[0]
        cats = _categorias_do_treino(X_tr)
        data = {c: (_como_category(X, cats), y) for c, (X, y) in data.items()}
    return data


# Treino e previsão
@dataclass
class FitResult:
    model: object
    melhor_iteracao: int
    n_arvores: int


def fit(X_tr, y_tr, X_va, y_va, config: XGBConfig = BASELINE, seed: int = cfg.SEED,
        device: str = "cpu") -> FitResult:
    """Treina com parada antecipada na log-loss de validação e guarda a melhor iteração."""
    from xgboost import XGBClassifier

    model = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", tree_method="hist", device=device,
        enable_categorical=True, n_estimators=config.max_arvores,
        early_stopping_rounds=config.paciencia, learning_rate=config.learning_rate,
        max_depth=config.max_depth, min_child_weight=config.min_child_weight,
        subsample=config.subsample, colsample_bytree=config.colsample_bytree, gamma=config.gamma,
        reg_lambda=config.reg_lambda, reg_alpha=config.reg_alpha, random_state=seed, n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return FitResult(model, int(model.best_iteration), int(model.best_iteration) + 1)


def predict_proba(model, X) -> np.ndarray:
    """Probabilidade de churn usando as árvores até a melhor iteração."""
    return model.predict_proba(X, iteration_range=(0, int(model.best_iteration) + 1))[:, 1]


def avaliar_configuracao(config: XGBConfig, variante: str, fold: int, seeds=(cfg.SEED,),
                         etapa: str = "E0_baseline", obs: str = "", incluir_teste: bool = True,
                         device: str = "cpu"):
    """Treina em cada semente e devolve (DataFrame no formato de `src.metrics`, saídas por semente)."""
    data = load_xy(fold, variante, cfg.SET_NAMES if incluir_teste else ("treino", "validacao"))
    (X_tr, y_tr), (X_va, y_va) = data["treino"], data["validacao"]
    frames, saidas = [], {}
    for s in seeds:
        r = fit(X_tr, y_tr, X_va, y_va, config, seed=s, device=device)
        scores = {c: (y, predict_proba(r.model, X)) for c, (X, y) in data.items()}
        hp = {**config_to_dict(config), "n_arvores": r.n_arvores}
        frames.append(evaluate_sets(MODELO, variante, fold, scores, hiperparametros=hp, obs=obs,
                                    etapa=etapa, semente=s))
        saidas[s] = {"fit": r, "scores": scores}
    return pd.concat(frames, ignore_index=True), saidas


# Busca de hiperparâmetros com Optuna
def suggest_config(trial) -> XGBConfig:
    """Espaço do slide de Gradient Boosting: taxa de aprendizagem, nº de árvores (parada antecipada),
    subsample, peso mínimo por folha, profundidade máxima, mais amostragem de colunas e regularização."""
    return XGBConfig(
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 2, 8),
        min_child_weight=trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        gamma=trial.suggest_float("gamma", 1e-3, 5.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    )


def make_objective(variante: str, fold: int = 1, seed: int = cfg.SEED, device: str = "cpu"):
    """Maximiza o KS de VALIDAÇÃO. Só treino e validação são lidos."""
    (X_tr, y_tr), (X_va, y_va) = load_xy(fold, variante).values()

    def objective(trial):
        c = suggest_config(trial)
        r = fit(X_tr, y_tr, X_va, y_va, c, seed=seed, device=device)
        p = predict_proba(r.model, X_va)
        trial.set_user_attr("val_auroc", float(roc_auc_score(y_va, p)))
        trial.set_user_attr("n_arvores", r.n_arvores)
        trial.set_user_attr("config", config_to_dict(c))
        return ks_score(y_va, p)

    return objective


def run_study(variante: str, n_trials: int = 100, fold: int = 1, seed: int = cfg.SEED,
              device: str = "cpu", storage_dir: Path = OPTUNA_DIR, mostrar_progresso: bool = True,
              timeout: float | None = None):
    """Cria (ou RETOMA) o estudo em SQLite e roda até completar `n_trials` tentativas no total."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    nome = f"{MODELO}_{variante}_fold{fold}"
    storage = f"sqlite:///{(storage_dir / (nome + '.db')).as_posix()}"
    try:
        feitas = len(optuna.load_study(study_name=nome, storage=storage).trials)
    except KeyError:
        feitas = 0
    study = optuna.create_study(study_name=nome, storage=storage, load_if_exists=True,
                                direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed + feitas))
    faltam = n_trials - len(study.trials)
    if faltam > 0:
        study.optimize(make_objective(variante, fold, seed, device), n_trials=faltam,
                       timeout=timeout, show_progress_bar=mostrar_progresso)
    return study


def trials_table(study) -> pd.DataFrame:
    """Tentativas concluídas (KS/AUROC de validação, nº de árvores, configuração), da melhor para a pior."""
    rows = [{"trial": t.number, "val_ks": t.value, "val_auroc": t.user_attrs["val_auroc"],
             "n_arvores": t.user_attrs["n_arvores"], **t.user_attrs["config"]}
            for t in study.trials if t.state.name == "COMPLETE"]
    return pd.DataFrame(rows).sort_values(["val_ks", "val_auroc"], ascending=False).reset_index(drop=True)


def reavaliar_candidatas(study, variante: str, k: int = 5, seeds=(42, 43, 44), fold: int = 1,
                         device: str = "cpu"):
    """Reavalia as k melhores tentativas com várias sementes (só validação) e escolhe a de maior KS médio."""
    top = trials_table(study).head(k)
    (X_tr, y_tr), (X_va, y_va) = load_xy(fold, variante).values()
    linhas, configs = [], {}
    for _, row in top.iterrows():
        c = config_from_dict(row.to_dict())
        ks, auc = [], []
        for s in seeds:
            p = predict_proba(fit(X_tr, y_tr, X_va, y_va, c, seed=s, device=device).model, X_va)
            ks.append(ks_score(y_va, p))
            auc.append(roc_auc_score(y_va, p))
        linhas.append({"trial": int(row["trial"]), "ks_trial_original": row["val_ks"],
                       "val_ks_media": float(np.mean(ks)), "val_ks_dp": float(np.std(ks, ddof=1)),
                       "val_auroc_media": float(np.mean(auc)), "sementes": len(ks)})
        configs[int(row["trial"])] = c
    tab = (pd.DataFrame(linhas).sort_values(["val_ks_media", "val_auroc_media"], ascending=False)
           .reset_index(drop=True))
    return tab, configs[int(tab.loc[0, "trial"])]


def export_trials(study, variante: str, fold: int = 1, out_dir: Path = cfg.REPORTS_TABLES) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"optuna_{MODELO}_{variante}_fold{fold}_trials.csv"
    trials_table(study).to_csv(path, index=False)
    return path


def etapa_busca(variante: str, etapa: str, n_trials: int = 100, fold: int = 1, k: int = 5,
                seeds_rerank=(42, 43, 44), seeds_registro=(42, 43, 44), device: str = "cpu",
                timeout: float | None = None, storage_dir: Path = OPTUNA_DIR,
                out_dir: Path = cfg.REPORTS_TABLES, mostrar_progresso: bool = True) -> dict:
    """Optuna (KS de validação) -> reavalia as k melhores com 3 sementes -> salva a escolhida
    (`xgboost_melhor_<variante>.json`) -> registra a etapa (o teste entra só no registro)."""
    from .metrics import save_results

    out_dir = Path(out_dir)
    study = run_study(variante, n_trials, fold, device=device, storage_dir=storage_dir,
                      mostrar_progresso=mostrar_progresso, timeout=timeout)
    export_trials(study, variante, fold, out_dir)
    ranking, melhor = reavaliar_candidatas(study, variante, k, seeds_rerank, fold, device)
    save_config(melhor, out_dir / f"{MODELO}_melhor_{variante}.json")
    obs = f"melhor de {len(study.trials)} tentativas Optuna; escolha por KS de validação"
    res, saidas = avaliar_configuracao(melhor, variante, fold, seeds_registro, etapa=etapa, obs=obs,
                                       device=device)
    save_results(res, out_dir)
    return {"study": study, "ranking": ranking, "config": melhor, "resultados": res, "saidas": saidas}


def melhor_etapa(results: pd.DataFrame, etapas, fold: int = 1):
    """Entre as `etapas`, a de maior KS MÉDIO de VALIDAÇÃO no fold (desempate AUROC). Nunca olha o teste."""
    d = results[(results["modelo"] == MODELO) & (results["fold"] == fold)
                & (results["conjunto"] == "validacao") & (results["etapa"].isin(list(etapas)))]
    if d.empty:
        raise ValueError(f"sem resultados de validação para {list(etapas)} no fold {fold}")
    tab = (d.groupby(["etapa", "variante"])[["ks", "auroc"]].mean().reset_index()
           .sort_values(["ks", "auroc"], ascending=False).reset_index(drop=True))
    return tab.loc[0, "etapa"], tab.loc[0, "variante"], tab


def etapa_final(config: XGBConfig, variante: str, etapa: str = "E3_final_3folds", folds=(1, 2, 3),
                seeds=(42, 43, 44), device: str = "cpu", out_dir: Path = cfg.REPORTS_TABLES):
    """Retreina a configuração escolhida em cada fold e semente e registra tudo."""
    from .metrics import save_results

    frames, saidas = [], {}
    for f in folds:
        res, s = avaliar_configuracao(config, variante, f, seeds, etapa=etapa, device=device,
                                      obs="configuração escolhida no Fold 1, retreinada neste fold")
        frames.append(res)
        saidas[f] = s
    df = pd.concat(frames, ignore_index=True)
    save_results(df, out_dir)
    return df, saidas


def importancia(model, top: int = 15) -> pd.Series:
    """Importância por ganho médio (quanto cada variável reduz a perda quando é usada numa divisão)."""
    imp = pd.Series(model.get_booster().get_score(importance_type="gain"))
    return imp.sort_values(ascending=False).head(top)


def plot_importancia(model, top: int = 15, titulo: str = "XGBoost: importância (ganho)", ax=None):
    import matplotlib.pyplot as plt

    imp = importancia(model, top)[::-1]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 0.35 * len(imp) + 1.2), layout="constrained")
    ax.barh(imp.index, imp.values, color="tab:blue")
    ax.set(title=titulo, xlabel="ganho médio")
    ax.grid(alpha=0.3, axis="x")
    return ax.figure
