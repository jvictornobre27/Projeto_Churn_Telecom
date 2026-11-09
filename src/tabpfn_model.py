"""Motor do TabPFN v2 (Transformer pré-treinado para dados tabulares, Prior Labs / Nature 2025).

O notebook `07_tabpfn.ipynb` só CHAMA este módulo. Diferenças em relação ao MLP/XGBoost:
- Não há treino por gradiente nem busca de hiperparâmetros: o TabPFN recebe o treino como CONTEXTO
  e prevê em uma passada (aprendizado em contexto). Por isso a escada varia a ENTRADA, não pesos.
- A versão do modelo é fixada em v2 (`ModelVersion.V2`), como pede o enunciado. O pacote atual usa
  versões mais novas por padrão (2.5+), que exigem aceitar licença; o v2 baixa sem token.
- A validação NÃO entra no modelo: serve só para escolher a etapa (KS de validação). O teste só é
  carregado com `incluir_teste=True`, para registro.
- KS/AUROC de TREINO são otimistas: o modelo "vê" as próprias linhas de treino no contexto.

Etapas possíveis (ver notebook): variante `neural` (one-hot + Min-Max) ou `arvore` (categóricas
nativas do TabPFN), contexto com ou sem as cópias do oversampling, e nº de membros do ensemble.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .metrics import evaluate_sets
from .preprocess import load_preprocessed

MODELO = "tabpfn"
VERSAO = "v2"
# Para testar a lógica sem baixar os pesos: atribuir uma função (config, seed, device, cat_idx) -> estimador
# com fit/predict_proba. Fica None no uso normal.
FABRICA_TESTE = None


@dataclass
class TabPFNConfig:
    n_estimators: int = 8           # membros do ensemble interno (permutações de colunas/pré-processamentos)
    sem_repeticao: bool = False     # contexto só com as linhas ORIGINAIS do treino (sem cópias do oversampling)
    softmax_temperature: float = 0.9   # padrão do v2


BASELINE = TabPFNConfig()


def config_to_dict(c: TabPFNConfig) -> dict:
    return {**asdict(c), "versao": VERSAO}


def config_from_dict(d: dict) -> TabPFNConfig:
    return TabPFNConfig(n_estimators=int(d.get("n_estimators", 8)),
                        sem_repeticao=bool(d.get("sem_repeticao", False)),
                        softmax_temperature=float(d.get("softmax_temperature", 0.9)))


def save_config(c: TabPFNConfig, path: Path, variante: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**config_to_dict(c), "variante": variante}, indent=2), encoding="utf-8")
    return path


def load_config(path: Path) -> tuple[TabPFNConfig, str]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return config_from_dict(d), d["variante"]


def device_padrao() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# Dados
def _codificar_categoricas(data: dict, X_tr_ref: pd.DataFrame) -> tuple[dict, list[int]]:
    """Variante `arvore`: categóricas viram códigos inteiros com categorias aprendidas SÓ no treino
    (valor novo em val/teste vira ausente, que o TabPFN trata). Devolve também os índices das categóricas."""
    cats = {c: sorted(X_tr_ref[c].astype(str).unique()) for c in cfg.CATEGORICAL_COLS if c in X_tr_ref.columns}
    out = {}
    for nome, (X, y) in data.items():
        X = X.copy()
        for c, valores in cats.items():
            X[c] = pd.Categorical(X[c].astype(str), categories=valores).codes.astype("float64")
            X.loc[X[c] < 0, c] = np.nan
        out[nome] = (X.astype("float64"), y)
    idx = [list(X_tr_ref.columns).index(c) for c in cats]
    return out, idx


def load_xy(fold: int, variante: str, conjuntos=("treino", "validacao")) -> tuple[dict, list[int] | None]:
    """({conjunto: (X, y)}, índices das categóricas ou None). Por padrão NÃO carrega o teste."""
    data = {c: load_preprocessed(fold, c, variante) for c in conjuntos}
    if variante == "arvore":
        X_tr = data["treino"][0] if "treino" in data else load_preprocessed(fold, "treino", variante)[0]
        return _codificar_categoricas(data, X_tr)
    return {c: (X.astype("float64"), y) for c, (X, y) in data.items()}, None


def mascara_originais(fold: int, conjunto: str = "treino") -> np.ndarray:
    """True nas linhas originais (False nas cópias criadas pelo oversampling), na ordem dos CSV do fold."""
    ids = pd.read_csv(cfg.DATA_PROCESSED / "ids_origem.csv")
    sel = ids[(ids["fold"] == fold) & (ids["conjunto"] == conjunto)]
    return ~sel["repetido"].to_numpy(dtype=bool)


# Modelo
def criar_modelo(config: TabPFNConfig, seed: int, device: str, cat_idx):
    if FABRICA_TESTE is not None:
        return FABRICA_TESTE(config, seed, device, cat_idx)
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    # Em CPU o pacote recusa mais de 1.000 linhas por padrão (só por lentidão). Nossa base (~3.450 linhas,
    # <= 61 colunas) está dentro dos limites do pré-treino do v2 (10.000 x 500), então liberamos só em CPU.
    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V2, n_estimators=config.n_estimators, softmax_temperature=config.softmax_temperature,
        categorical_features_indices=cat_idx, random_state=seed, device=device,
        ignore_pretraining_limits=(str(device) == "cpu"))


def fit(X_tr, y_tr, config: TabPFNConfig = BASELINE, seed: int = cfg.SEED, device: str | None = None,
        cat_idx=None, mask_originais=None):
    """'Treina' = guarda o contexto. Com `sem_repeticao`, usa só as linhas originais do treino."""
    device = device or device_padrao()
    if config.sem_repeticao:
        if mask_originais is None:
            raise ValueError("sem_repeticao=True precisa de mask_originais")
        X_tr, y_tr = X_tr[mask_originais], y_tr[mask_originais]
    model = criar_modelo(config, seed, device, cat_idx)
    model.fit(np.asarray(X_tr), np.asarray(y_tr))
    return model


def predict_proba(model, X) -> np.ndarray:
    return model.predict_proba(np.asarray(X))[:, 1]


def avaliar_configuracao(config: TabPFNConfig, variante: str, fold: int, seeds=(cfg.SEED,),
                         etapa: str = "E0_baseline", obs: str = "", incluir_teste: bool = True,
                         device: str | None = None):
    """Ajusta em cada semente e devolve (DataFrame no formato de `src.metrics`, saídas por semente).

    As métricas de treino são sempre calculadas no treino OFICIAL do fold (com repetições), para
    ficarem comparáveis às dos outros modelos, mesmo quando o contexto não usa as cópias.
    """
    data, cat_idx = load_xy(fold, variante, cfg.SET_NAMES if incluir_teste else ("treino", "validacao"))
    X_tr, y_tr = data["treino"]
    mask = mascara_originais(fold) if config.sem_repeticao else None
    if mask is not None and len(mask) != len(X_tr):
        raise AssertionError("ids_origem.csv não bate com o treino do fold")
    frames, saidas = [], {}
    for s in seeds:
        model = fit(X_tr.to_numpy(), y_tr.to_numpy(), config, seed=s, device=device, cat_idx=cat_idx,
                    mask_originais=mask)
        scores = {c: (y, predict_proba(model, X)) for c, (X, y) in data.items()}
        hp = {**config_to_dict(config), "linhas_contexto": int(mask.sum()) if mask is not None else len(X_tr)}
        frames.append(evaluate_sets(MODELO, variante, fold, scores, hiperparametros=hp, obs=obs,
                                    etapa=etapa, semente=s))
        saidas[s] = {"model": model, "scores": scores}
    return pd.concat(frames, ignore_index=True), saidas


def melhor_etapa(results: pd.DataFrame, etapas, fold: int = 1):
    """Entre as `etapas`, a de maior KS MÉDIO de VALIDAÇÃO no fold (desempate AUROC). Nunca olha o teste."""
    d = results[(results["modelo"] == MODELO) & (results["fold"] == fold)
                & (results["conjunto"] == "validacao") & (results["etapa"].isin(list(etapas)))]
    if d.empty:
        raise ValueError(f"sem resultados de validação para {list(etapas)} no fold {fold}")
    tab = (d.groupby(["etapa", "variante"])[["ks", "auroc"]].mean().reset_index()
           .sort_values(["ks", "auroc"], ascending=False).reset_index(drop=True))
    return tab.loc[0, "etapa"], tab.loc[0, "variante"], tab


def config_da_etapa(results: pd.DataFrame, etapa: str, fold: int = 1) -> TabPFNConfig:
    """Lê a configuração registrada de uma etapa (coluna `hiperparametros`)."""
    d = results[(results["modelo"] == MODELO) & (results["etapa"] == etapa) & (results["fold"] == fold)]
    return config_from_dict(json.loads(d["hiperparametros"].iloc[0]))


def etapa_final(config: TabPFNConfig, variante: str, etapa: str = "E4_final_3folds", folds=(1, 2, 3),
                seeds=(42, 43, 44), device: str | None = None, out_dir: Path = cfg.REPORTS_TABLES):
    """Reaplica a configuração escolhida em cada fold e semente e registra tudo."""
    from .metrics import save_results

    frames, saidas = [], {}
    for f in folds:
        res, s = avaliar_configuracao(config, variante, f, seeds, etapa=etapa, device=device,
                                      obs="configuração escolhida no Fold 1, reaplicada neste fold")
        frames.append(res)
        saidas[f] = s
    df = pd.concat(frames, ignore_index=True)
    save_results(df, out_dir)
    return df, saidas
