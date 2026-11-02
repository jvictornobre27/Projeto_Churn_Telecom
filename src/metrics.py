"""Métricas de avaliação compartilhadas por TODOS os modelos.

Por que existe: se cada pessoa calcular o KS (ou o limiar de classificação) de um
jeito, os resultados deixam de ser comparáveis. Todos os notebooks de modelo devem
usar estas funções e salvar os resultados no formato daqui.

Convenções:
- `y_true`  : 0/1 (1 = churn), como em `y_*_fold*.csv`.
- `y_score` : PROBABILIDADE da classe 1, entre 0 e 1 (vetor 1-D, ou matriz (n, 2) de
  `predict_proba`, de onde se usa a coluna 1). Se o modelo devolve logits, aplique
  sigmoid/softmax antes; valores fora de [0, 1] geram erro de propósito.
- KS (principal) = max(TPR - FPR) = maior distância entre as
  distribuições acumuladas dos scores de churn e de não-churn. Vai de 0 a 1.
- `mse` é o erro quadrático médio entre a probabilidade e o rótulo (Brier score).
- Métricas de classe (acurácia, precisão, recall, especificidade, F1) usam `limiar` (padrão 0,5).
  KS, AUROC, AUPR, MSE e log_loss NÃO dependem do limiar.
- Matriz de confusão: o PADRÃO É PROPORCIONAL POR LINHA (cada linha dividida pelo total da
  classe real), então a diagonal é a especificidade (classe 0) e o recall (classe 1) e a
  matriz é comparável entre treino, validação e teste mesmo com balanceamentos diferentes.
  As contagens nominais (tn, fp, fn, tp) continuam salvas no CSV; `plot_confusion` aceita
  `normalize="nominal"` quando precisar delas.
- Treino/validação são balanceados (50%/50%) e o teste não (~26,5% de churn):
  precisão e F1 mudam com a proporção; KS e AUROC não. Por isso `pct_positivos`
  é sempre registrado.

Fluxo típico em um notebook de modelo:
    from src.metrics import evaluate_sets, save_results
    res = evaluate_sets("mlp", "neural", fold=1, scores={
        "treino":    (y_tr, p_tr),
        "validacao": (y_va, p_va),
        "teste":     (y_te, p_te),      # só na avaliação final, nunca para escolher hiperparâmetros
    }, hiperparametros={"camadas": 1, "neuronios": 10},
       etapa="E0_baseline", semente=42)
    save_results(res)                   # -> reports/tables/resultados_mlp.csv

Evolução: cada degrau do modelo recebe uma `etapa` ("E0_baseline", "E1_faixas", ...).
`plot_evolution(load_results(), "mlp")` mostra o KS por etapa; a escolha da etapa/configuração
usa SÓ a validação. Sementes: salve uma linha por semente; `summarize_folds` tira a média das
sementes dentro de cada fold antes de calcular média e desvio entre os folds.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

from . import config as cfg

DEFAULT_THRESHOLD = 0.5
METRIC_COLS = ["n", "pct_positivos", "ks", "ks_limiar", "auroc", "aupr", "mse", "log_loss",
               "acuracia", "precisao", "recall", "especificidade", "f1", "limiar", "tn", "fp", "fn", "tp"]
# `etapa` = degrau da escada evolutiva do modelo (ex.: "E0_baseline", "E1_faixas"); o prefixo
# "E<n>" define a ordem nos gráficos. `semente` = semente de inicialização/treino do modelo.
KEY_COLS = ["modelo", "etapa", "variante", "fold", "semente", "conjunto"]
DEFAULT_ETAPA = "E0_baseline"


# Validação de entrada
def _prepare(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).ravel()
    s = np.asarray(y_score, dtype=float)
    if s.ndim == 2:
        if s.shape[1] != 2:
            raise ValueError(f"y_score 2-D deve ter 2 colunas (predict_proba); recebeu {s.shape}")
        s = s[:, 1]
    s = s.ravel()
    if len(y) != len(s):
        raise ValueError(f"y_true ({len(y)}) e y_score ({len(s)}) com tamanhos diferentes")
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError("y_true deve conter apenas 0 e 1")
    if len(np.unique(y)) < 2:
        raise ValueError("y_true tem uma única classe; KS/AUROC não são definidos")
    if np.isnan(s).any():
        raise ValueError("y_score contém NaN")
    if s.min() < -1e-9 or s.max() > 1 + 1e-9:
        raise ValueError("y_score deve ser probabilidade em [0, 1]; se o modelo devolve logits, "
                         "aplique sigmoid/softmax antes")
    return y.astype(int), np.clip(s, 0.0, 1.0)


# Métricas
def ks_statistic(y_true, y_score) -> tuple[float, float]:
    """Retorna (KS, limiar de score onde a separação é máxima)."""
    y, s = _prepare(y_true, y_score)
    fpr, tpr, thr = roc_curve(y, s)
    d = tpr - fpr
    i = int(np.argmax(d))
    return float(d[i]), (float(thr[i]) if np.isfinite(thr[i]) else float("nan"))


def ks_score(y_true, y_score) -> float:
    """Só o valor do KS (atalho para usar como objetivo no Optuna)."""
    return ks_statistic(y_true, y_score)[0]


def compute_metrics(y_true, y_score, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Todas as métricas do projeto para um conjunto (um fold, um conjunto)."""
    y, s = _prepare(y_true, y_score)
    ks, ks_thr = ks_statistic(y, s)
    pred = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "pct_positivos": round(100 * float(y.mean()), 2),
        "ks": ks,
        "ks_limiar": ks_thr,
        "auroc": float(roc_auc_score(y, s)),
        "aupr": float(average_precision_score(y, s)),
        "mse": float(np.mean((s - y) ** 2)),
        "log_loss": float(log_loss(y, np.clip(s, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "acuracia": float(accuracy_score(y, pred)),
        "precisao": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "especificidade": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "limiar": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# Resultados: formato único + salvar/carregar/resumir
def evaluate_sets(
    modelo: str,
    variante: str,
    fold: int,
    scores: dict[str, tuple],
    threshold: float = DEFAULT_THRESHOLD,
    hiperparametros: dict | None = None,
    obs: str = "",
    etapa: str = DEFAULT_ETAPA,
    semente: int = cfg.SEED,
) -> pd.DataFrame:
    """Calcula as métricas de cada conjunto e devolve um DataFrame no formato do projeto.

    `scores` = {"treino": (y, p), "validacao": (y, p), "teste": (y, p)} (qualquer subconjunto).
    `etapa` = degrau da evolução (ver `plot_evolution`); `semente` = semente do treino do modelo.
    """
    unknown = set(scores) - set(cfg.SET_NAMES)
    if unknown:
        raise ValueError(f"Conjuntos inválidos: {sorted(unknown)}; use {cfg.SET_NAMES}")
    hp = json.dumps(hiperparametros or {}, sort_keys=True, ensure_ascii=False, default=str)
    rows = []
    for name in cfg.SET_NAMES:
        if name not in scores:
            continue
        y, p = scores[name]
        rows.append({"modelo": _slug(modelo), "etapa": etapa, "variante": variante, "fold": int(fold),
                     "semente": int(semente), "conjunto": name,
                     "hiperparametros": hp, "obs": obs, **compute_metrics(y, p, threshold)})
    return pd.DataFrame(rows)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", name.strip().lower()).strip("_")


def save_results(df: pd.DataFrame, out_dir: Path = cfg.REPORTS_TABLES) -> Path:
    """Grava em `<out_dir>/resultados_<modelo>.csv`.

    Se já existir linha com a mesma chave (modelo, etapa, variante, fold, semente, conjunto),
    ela é SUBSTITUÍDA, então rodar o notebook de novo não duplica resultados.
    Arquivos antigos sem `etapa`/`semente` são lidos com os valores padrão.
    """
    modelos = df["modelo"].unique()
    if len(modelos) != 1:
        raise ValueError(f"save_results espera um único modelo por chamada; recebeu {list(modelos)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"resultados_{modelos[0]}.csv"
    if path.exists():
        old = _fill_legacy(pd.read_csv(path))
        keys_new = set(map(tuple, df[KEY_COLS].astype(str).to_numpy()))
        keep = ~old[KEY_COLS].astype(str).apply(tuple, axis=1).isin(keys_new)
        df = pd.concat([old[keep], df], ignore_index=True)
    order = {n: i for i, n in enumerate(cfg.SET_NAMES)}
    df = (df.assign(_o=df["conjunto"].map(order))
            .sort_values(["etapa", "variante", "fold", "semente", "_o"]).drop(columns="_o")
            .reset_index(drop=True))
    df.to_csv(path, index=False)
    return path


def _fill_legacy(df: pd.DataFrame) -> pd.DataFrame:
    """Garante as colunas `etapa` e `semente` em tabelas salvas antes de elas existirem."""
    if "etapa" not in df.columns:
        df["etapa"] = DEFAULT_ETAPA
    if "semente" not in df.columns:
        df["semente"] = cfg.SEED
    return df


def load_results(out_dir: Path = cfg.REPORTS_TABLES) -> pd.DataFrame:
    """Junta todos os `resultados_*.csv` (um por modelo) em uma tabela só."""
    files = sorted(out_dir.glob("resultados_*.csv"))
    return _fill_legacy(pd.concat([pd.read_csv(f) for f in files], ignore_index=True)) if files else pd.DataFrame()


def _fold_means(d: pd.DataFrame, metrics) -> pd.DataFrame:
    """Média das SEMENTES dentro de cada fold (uma linha por modelo/etapa/variante/fold)."""
    d = _fill_legacy(d.copy())
    g = d.groupby(["modelo", "etapa", "variante", "fold"])
    out = g[list(metrics)].mean()
    out["n_sementes"] = g["semente"].nunique()
    # desvio entre sementes dentro do fold (0 se há uma semente só)
    out["ks_dp_sementes"] = g["ks"].std(ddof=1).fillna(0.0)
    return out.reset_index()


def summarize_folds(
    results: pd.DataFrame,
    conjunto: str = "teste",
    metrics: tuple = ("ks", "auroc", "f1", "recall", "precisao", "mse"),
    n_folds: int = cfg.K_FOLDS,
) -> pd.DataFrame:
    """Média e desvio padrão (amostral, ddof=1) por modelo/etapa/variante ao longo dos folds, ordenado por KS.

    Primeiro faz a média das sementes dentro de cada fold; o desvio entre folds é calculado
    SOBRE essas médias (sementes não inflam o número de folds). `ks_dp_sementes` mostra
    separadamente a variação por semente (média dos desvios dentro dos folds).
    A coluna `folds` mostra quantos folds existem; menos que `n_folds` = comparação incompleta.
    """
    fm = _fold_means(results[results["conjunto"] == conjunto], metrics)
    g = fm.groupby(["modelo", "etapa", "variante"])
    out = g[list(metrics)].agg(["mean", "std"])
    out.columns = [f"{m}_{'media' if s == 'mean' else 'dp'}" for m, s in out.columns]
    out["ks_dp_sementes"] = g["ks_dp_sementes"].mean()
    out["sementes"] = g["n_sementes"].max()
    out["folds"] = g["fold"].nunique()
    out["completo"] = out["folds"] >= n_folds
    return out.reset_index().sort_values("ks_media", ascending=False).reset_index(drop=True)


def evolution_table(results: pd.DataFrame, modelo: str, fold: int | None = None,
                    metrics: tuple = ("ks", "auroc", "mse", "f1")) -> pd.DataFrame:
    """Tabela da evolução de um modelo: uma linha por etapa e conjunto (média das sementes).

    `fold=None` usa a média dos folds disponíveis na etapa; `fold=1` mostra só aquele fold
    (a escada de busca é feita no Fold 1).
    """
    d = _fill_legacy(results[results["modelo"] == _slug(modelo)].copy())
    if fold is not None:
        d = d[d["fold"] == fold]
    if d.empty:
        raise ValueError(f"Sem resultados para modelo={modelo!r}, fold={fold}")
    out = (d.groupby(["etapa", "variante", "conjunto"])[list(metrics)].mean().reset_index())
    out["folds"] = d.groupby(["etapa", "variante", "conjunto"])["fold"].nunique().values
    order = {n: i for i, n in enumerate(cfg.SET_NAMES)}
    return (out.assign(_o=out["conjunto"].map(order)).sort_values(["etapa", "_o"])
               .drop(columns="_o").reset_index(drop=True))


# Gráficos de evolução de métricas
def plot_evolution(results: pd.DataFrame, modelo: str, metric: str = "ks", fold: int | None = 1,
                   conjuntos: tuple = ("validacao", "teste"), ax=None):
    """Evolução de UM modelo ao longo das etapas (E0, E1, ...): uma linha por conjunto.

    Barras de erro = desvio entre sementes (quando há mais de uma). A ESCOLHA da etapa/configuração
    deve olhar só a validação; o teste é mostrado ao lado apenas para registro.
    """
    import matplotlib.pyplot as plt

    d = _fill_legacy(results[results["modelo"] == _slug(modelo)].copy())
    if fold is not None:
        d = d[d["fold"] == fold]
    if d.empty:
        raise ValueError(f"Sem resultados para modelo={modelo!r}, fold={fold}")
    etapas = sorted(d["etapa"].unique())
    if ax is None:
        _, ax = plt.subplots(figsize=(max(5.5, 1.6 * len(etapas)), 4), layout="constrained")
    cores = {"treino": "tab:gray", "validacao": "tab:blue", "teste": "tab:red"}
    for conj in conjuntos:
        s = d[d["conjunto"] == conj]
        if s.empty:
            continue
        # média das sementes dentro do fold, depois média dos folds; desvio entre sementes
        por_fold = s.groupby(["etapa", "fold"])[metric].agg(["mean", "std"]).fillna(0.0)
        g = por_fold.groupby("etapa").mean().reindex(etapas)
        ax.errorbar(range(len(etapas)), g["mean"], yerr=g["std"], marker="o", capsize=3,
                    label=conj, color=cores.get(conj))
        for i, v in enumerate(g["mean"]):
            if np.isfinite(v):
                ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points", xytext=(0, 7),
                            ha="center", fontsize=8, color=cores.get(conj))
    ax.set_xticks(range(len(etapas)))
    ax.set_xticklabels([e.replace("_", "\n", 1) for e in etapas], fontsize=8)
    escopo = f"fold {fold}" if fold is not None else "média dos folds"
    ax.set(title=f"Evolução de {modelo}: {metric.upper()} por etapa ({escopo})", ylabel=metric.upper())
    ax.grid(alpha=0.3)
    ax.legend()
    return ax.figure


def plot_ks(y_true, y_score, title: str = "KS", ax=None):
    """Distribuições acumuladas dos scores de churn e não-churn, com o KS marcado."""
    import matplotlib.pyplot as plt

    y, s = _prepare(y_true, y_score)
    fpr, tpr, thr = roc_curve(y, s)
    keep = np.isfinite(thr)
    x, cdf_neg, cdf_pos = thr[keep][::-1], (1 - fpr)[keep][::-1], (1 - tpr)[keep][::-1]
    ks, ks_thr = ks_statistic(y, s)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(x, cdf_neg, label="Não churn (0)", color="tab:blue")
    ax.plot(x, cdf_pos, label="Churn (1)", color="tab:red")
    if np.isfinite(ks_thr):
        yi = np.interp(ks_thr, x, cdf_neg)
        ax.vlines(ks_thr, yi - ks, yi, colors="k", linestyles="--")
    ax.set(title=f"{title}  (KS = {ks:.3f})", xlabel="Score (prob. de churn)", ylabel="Distribuição acumulada")
    ax.legend(loc="lower right")
    return ax.figure


def plot_roc(y_true, y_score, title: str = "ROC", ax=None):
    import matplotlib.pyplot as plt

    y, s = _prepare(y_true, y_score)
    fpr, tpr, _ = roc_curve(y, s)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, color="tab:red", label=f"AUROC = {roc_auc_score(y, s):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(title=title, xlabel="Falso positivo", ylabel="Verdadeiro positivo")
    ax.legend(loc="lower right")
    return ax.figure


def confusion_proportions(y_true, y_score, threshold: float = DEFAULT_THRESHOLD, normalize: str = "true") -> np.ndarray:
    """Matriz de confusão 2x2 (linhas = real 0/1, colunas = previsto 0/1).

    normalize:
      "true"    (PADRÃO) cada linha soma 1: diagonal = especificidade e recall; comparável entre conjuntos.
      "pred"    cada coluna soma 1: diagonal = valor preditivo negativo e precisão (depende do balanceamento).
      "all"     dividida pelo total (depende do balanceamento).
      "nominal" contagens de linhas.
    """
    if normalize not in ("true", "pred", "all", "nominal"):
        raise ValueError("normalize deve ser 'true', 'pred', 'all' ou 'nominal'")
    y, s = _prepare(y_true, y_score)
    cm = confusion_matrix(y, (s >= threshold).astype(int), labels=[0, 1])
    if normalize == "nominal":
        return cm
    return confusion_matrix(y, (s >= threshold).astype(int), labels=[0, 1], normalize=normalize)


def plot_confusion(y_true, y_score, threshold: float = DEFAULT_THRESHOLD, normalize: str = "true",
                   title: str = "Matriz de confusão", ax=None):
    """Matriz de confusão; por padrão PROPORCIONAL POR LINHA (ver `confusion_proportions`)."""
    import matplotlib.pyplot as plt

    cm = confusion_proportions(y_true, y_score, threshold, normalize)
    nominal = normalize == "nominal"
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 4.2), layout="constrained")
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() if nominal else 1)
    for (i, j), v in np.ndenumerate(cm):
        txt = str(v) if nominal else f"{100 * v:.1f}%"
        ax.text(j, i, txt, ha="center", va="center", color="white" if v > (cm.max() if nominal else 1) / 2 else "black")
    suffix = {"true": "proporção por classe real", "pred": "proporção por classe prevista",
              "all": "proporção do total", "nominal": "contagens"}[normalize]
    ax.set(title=f"{title}\n(limiar {threshold:g}; {suffix})", xlabel="Previsto", ylabel="Real",
           xticks=[0, 1], yticks=[0, 1], xticklabels=["0", "1"], yticklabels=["0", "1"])
    return ax.figure


def save_figure(fig, name: str, out_dir: Path = cfg.REPORTS_FIGURES) -> Path:
    """Salva a figura em reports/figures/<name>.png (nomeie com o modelo, ex.: 'mlp_ks_fold1_teste')."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{_slug(name)}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
