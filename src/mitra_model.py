"""Motor do Mitra (modelo de fundação tabular da Amazon / AutoGluon, pré-treinado em dados sintéticos).

O notebook `08_mitra.ipynb` só CHAMA este módulo. Como o TabPFN, o Mitra é um Transformer que usa o
treino como CONTEXTO (aprendizado em contexto); opcionalmente pode ser ajustado (`fine_tune=True`).
Por isso a escada varia a ENTRADA e o modo (zero-shot x fine-tuning), não uma busca de hiperparâmetros.

Interface: `TabularPredictor` do AutoGluon com `hyperparameters={"MITRA": {...}}`, como na model card
do Hugging Face (autogluon/mitra-classifier e autogluon/mitra-classifier-2).

Regras iguais às dos outros modelos:
- A validação serve só para ESCOLHER a etapa (KS de validação). O teste só é carregado com `incluir_teste=True`.
- KS/AUROC de TREINO são otimistas: o modelo "vê" as próprias linhas de treino no contexto.

AVISO: escrito sem poder rodar o Mitra no ambiente de desenvolvimento. O primeiro teste é no Colab
(`autogluon.tabular[mitra]`). Se a API do AutoGluon reclamar de algum argumento, o ponto de ajuste é `fit`.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .metrics import evaluate_sets
from .preprocess import load_preprocessed
from .tabpfn_model import mascara_originais   # mesma máscara de linhas originais (sem cópias do oversampling)

MODELO = "mitra"
LABEL = "__churn__"
# O AutoGluon estima ~12,5 GB de RAM para o contexto com oversampling (3.450 linhas x 61 colunas) e recusa treinar
# no Colab gratuito (~10,7 GB livres). A estimativa é conservadora; liberamos o limite (o erro sugere >= 1,34).
MAX_MEMORY_RATIO = 1.7
CHECKPOINT_V1 = "autogluon/mitra-classifier"
CHECKPOINT_V2 = "autogluon/mitra-classifier-2"   # exige autogluon.tabular>=1.6 e o pacote mitra-finetune
# Para testar a lógica sem baixar pesos: função (config, seed) -> objeto com fit(X, y, X_va, y_va) e predict_proba(X).
FABRICA_TESTE = None


@dataclass
class MitraConfig:
    hf_model: str = CHECKPOINT_V1
    fine_tune: bool = False        # False = zero-shot (aprendizado em contexto); True = ajusta os pesos no treino
    sem_repeticao: bool = False    # contexto só com as linhas ORIGINAIS do treino (sem cópias do oversampling)


BASELINE = MitraConfig()


def config_to_dict(c: MitraConfig) -> dict:
    return asdict(c)


def config_from_dict(d: dict) -> MitraConfig:
    return MitraConfig(hf_model=str(d.get("hf_model", CHECKPOINT_V1)), fine_tune=bool(d.get("fine_tune", False)),
                       sem_repeticao=bool(d.get("sem_repeticao", False)))


def save_config(c: MitraConfig, path: Path, variante: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**config_to_dict(c), "variante": variante}, indent=2), encoding="utf-8")
    return path


def device_padrao() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_xy(fold: int, variante: str, conjuntos=("treino", "validacao")) -> dict:
    """{conjunto: (X, y)}. No Mitra a variante `arvore` (categóricas como texto) é aceita pelo AutoGluon."""
    return {c: load_preprocessed(fold, c, variante) for c in conjuntos}


class _Modelo:
    """Embrulha o TabularPredictor para ter fit/predict_proba como os demais modelos."""

    def __init__(self, config: MitraConfig, seed: int):
        self.config, self.seed, self.pred, self.dir = config, seed, None, None

    def fit(self, X_tr, y_tr, X_va, y_va):
        from autogluon.tabular import TabularPredictor

        self.dir = tempfile.mkdtemp(prefix="mitra_")
        tr = X_tr.copy()
        tr[LABEL] = np.asarray(y_tr).astype(int)
        va = X_va.copy()
        va[LABEL] = np.asarray(y_va).astype(int)
        hp = {"MITRA": {"hf_model": self.config.hf_model, "fine_tune": bool(self.config.fine_tune),
                         "ag.max_memory_usage_ratio": MAX_MEMORY_RATIO}}
        kw = {"num_gpus": 1} if device_padrao() == "cuda" else {}
        self.pred = TabularPredictor(label=LABEL, problem_type="binary", eval_metric="roc_auc",
                                     path=self.dir, verbosity=1)
        # tuning_data = validação oficial do fold (nunca o teste). Sem ensemble ponderado: só o Mitra.
        self.pred.fit(tr, tuning_data=va, hyperparameters=hp, fit_weighted_ensemble=False,
                      time_limit=None, **kw)
        return self

    def predict_proba(self, X) -> np.ndarray:
        p = self.pred.predict_proba(X, as_multiclass=False)
        return np.asarray(p, dtype=float).ravel()

    def limpar(self):
        """Libera RAM/GPU: apaga a pasta temporária e solta o preditor (o modelo fica carregado até ser coletado)."""
        if self.dir:
            shutil.rmtree(self.dir, ignore_errors=True)
        self.pred = None
        liberar_memoria()


def liberar_memoria() -> None:
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def criar_modelo(config: MitraConfig, seed: int):
    if FABRICA_TESTE is not None:
        return FABRICA_TESTE(config, seed)
    return _Modelo(config, seed)


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def avaliar_configuracao(config: MitraConfig, variante: str, fold: int, seeds=(cfg.SEED,),
                         etapa: str = "E0_baseline", obs: str = "", incluir_teste: bool = True):
    """Ajusta em cada semente e devolve (DataFrame no formato de `src.metrics`, saídas por semente).

    As métricas de treino são sempre calculadas no treino OFICIAL do fold (com repetições), para ficarem
    comparáveis às dos outros modelos, mesmo quando o contexto não usa as cópias.
    """
    data = load_xy(fold, variante, cfg.SET_NAMES if incluir_teste else ("treino", "validacao"))
    (X_tr, y_tr), (X_va, y_va) = data["treino"], data["validacao"]
    mask = None
    if config.sem_repeticao:
        mask = mascara_originais(fold)
        if len(mask) != len(X_tr):
            raise AssertionError("ids_origem.csv não bate com o treino do fold")
    frames, saidas = [], {}
    for s in seeds:
        _set_seed(s)
        Xc, yc = (X_tr[mask], y_tr[mask]) if mask is not None else (X_tr, y_tr)
        model = criar_modelo(config, s).fit(Xc.reset_index(drop=True), yc.reset_index(drop=True), X_va, y_va)
        scores = {c: (y, model.predict_proba(X)) for c, (X, y) in data.items()}
        hp = {**config_to_dict(config), "linhas_contexto": int(len(Xc))}
        frames.append(evaluate_sets(MODELO, variante, fold, scores, hiperparametros=hp, obs=obs,
                                    etapa=etapa, semente=s))
        # salva já a cada semente: se a sessão cair na seguinte, esta não se perde
        from .metrics import save_results
        save_results(frames[-1], cfg.REPORTS_TABLES)
        saidas[s] = {"scores": scores}          # NÃO guarda o modelo (ocupa RAM e a próxima semente falha)
        if hasattr(model, "limpar"):
            model.limpar()
        del model
        liberar_memoria()
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


def config_da_etapa(results: pd.DataFrame, etapa: str, fold: int = 1) -> MitraConfig:
    d = results[(results["modelo"] == MODELO) & (results["etapa"] == etapa) & (results["fold"] == fold)]
    return config_from_dict(json.loads(d["hiperparametros"].iloc[0]))


def etapa_final(config: MitraConfig, variante: str, etapa: str = "E4_final_3folds", folds=(1, 2, 3),
                seeds=(42, 43, 44), out_dir: Path = cfg.REPORTS_TABLES):
    """Reaplica a configuração escolhida em cada fold e semente e registra tudo."""
    from .metrics import save_results

    frames, saidas = [], {}
    for f in folds:
        res, s = avaliar_configuracao(config, variante, f, seeds, etapa=etapa,
                                      obs="configuração escolhida no Fold 1, reaplicada neste fold")
        frames.append(res)
        saidas[f] = s
    df = pd.concat(frames, ignore_index=True)
    save_results(df, out_dir)
    return df, saidas
