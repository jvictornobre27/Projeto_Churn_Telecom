"""Pré-processamento por fold, sem vazamento de dados.

Regra central: tudo que é *aprendido* dos dados (categorias do one-hot,
min/max, mediana de contingência, quartis das faixas) é ajustado **somente no
treino** do fold e depois aplicado, sem recalcular, em validação e teste.

Detalhes de projeto:
- `fit` recebe apenas o treino. Quando o treino tem linhas repetidas pelo
  oversampling, os parâmetros são ajustados nas linhas **originais**
  (`repetido == False`), para que a repetição não distorça estatísticas.
- `TotalCharges` ausente: todos os casos têm `tenure == 0` (cliente que ainda não
  foi cobrado), então a regra é imputar 0. Se surgir outro ausente, usa-se a
  mediana do treino (fallback).
- Validação/teste NÃO são alterados em número de linhas (nada é removido) nem
  recortados (valores fora do intervalo do treino podem ficar fora de [0, 1]).

Uso pela linha de comando:
    python -m src.preprocess
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from . import config as cfg
from .split_data import load_ids, load_partition

log = logging.getLogger("preprocess")

BIN_COLS = ["tenure_faixa", "MonthlyCharges_faixa"]


class Preprocessor:
    """Ajusta no treino, transforma qualquer conjunto."""

    def __init__(self, onehot: bool = True, scale: bool = True, bins: bool = False):
        self.onehot, self.scale, self.bins = onehot, scale, bins

    # ---- passos determinísticos (não aprendem nada, exceto o fallback da mediana)
    def _impute(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        zero = X["TotalCharges"].isna() & (X["tenure"] == 0)
        X.loc[zero, "TotalCharges"] = 0.0
        rest = X["TotalCharges"].isna()
        if rest.any():
            X.loc[rest, "TotalCharges"] = self.total_median_
        return X

    def _add_bins(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        t_edges = [-np.inf, *cfg.TENURE_BIN_EDGES, np.inf]
        X["tenure_faixa"] = pd.cut(X["tenure"], t_edges, labels=cfg.TENURE_BIN_LABELS).astype(str)
        m_edges = [-np.inf, *self.monthly_edges_, np.inf]
        X["MonthlyCharges_faixa"] = pd.cut(X["MonthlyCharges"], m_edges, labels=cfg.MONTHLY_BIN_LABELS).astype(str)
        return X

    def _cat_cols(self) -> list[str]:
        return cfg.CATEGORICAL_COLS + (BIN_COLS if self.bins else [])

    def _cat_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self._cat_cols()].astype(str)

    def fit(self, X_train: pd.DataFrame, original_mask: np.ndarray | None = None) -> "Preprocessor":
        base = X_train if original_mask is None else X_train[original_mask]
        # mediana de contingência (usada só se houver ausente com tenure > 0)
        pos = base.loc[~((base["TotalCharges"].isna()) & (base["tenure"] == 0)), "TotalCharges"]
        self.total_median_ = float(pos.median())
        base = self._impute(base)
        self.monthly_edges_ = [float(q) for q in base["MonthlyCharges"].quantile(cfg.MONTHLY_BIN_QUANTILES)]
        if self.bins:
            base = self._add_bins(base)
        if self.scale:
            self.scaler_ = MinMaxScaler().fit(base[cfg.NUMERIC_COLS])
        if self.onehot:
            self.ohe_ = OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)
            self.ohe_.fit(self._cat_frame(base))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._impute(X)
        if self.bins:
            X = self._add_bins(X)
        if not self.onehot and not self.scale:       # variante para árvores
            cols = cfg.NUMERIC_COLS + self._cat_cols()
            return X[cols].reset_index(drop=True)
        num = X[cfg.NUMERIC_COLS].reset_index(drop=True)
        if self.scale:
            num = pd.DataFrame(self.scaler_.transform(num), columns=cfg.NUMERIC_COLS)
        if not self.onehot:
            return pd.concat([num, X[self._cat_cols()].reset_index(drop=True)], axis=1)
        cat = pd.DataFrame(self.ohe_.transform(self._cat_frame(X)),
                           columns=self.ohe_.get_feature_names_out(self._cat_cols()))
        return pd.concat([num, cat], axis=1)

    def unknown_categories(self, X: pd.DataFrame) -> int:
        """Quantos valores categóricos de X não existiam no treino (one-hot ignora)."""
        if not self.onehot:
            return 0
        X = self._impute(X)
        X = self._add_bins(X) if self.bins else X
        cats = self._cat_frame(X)
        n = 0
        for col, known in zip(self._cat_cols(), self.ohe_.categories_):
            n += int((~cats[col].isin(known)).sum())
        return n

    # auditoria
    def params(self) -> dict:
        p = {"onehot": self.onehot, "scale": self.scale, "bins": self.bins,
             "TotalCharges_mediana_contingencia": self.total_median_,
             "regra_TotalCharges": "tenure == 0 -> 0.0",
             "MonthlyCharges_quartis": self.monthly_edges_}
        if self.scale:
            p["minmax"] = {c: {"min": float(a), "max": float(b)} for c, a, b in
                           zip(cfg.NUMERIC_COLS, self.scaler_.data_min_, self.scaler_.data_max_)}
        if self.onehot:
            p["categorias"] = {c: [str(v) for v in k] for c, k in zip(self._cat_cols(), self.ohe_.categories_)}
        return p


# Execução por fold e variante
def variant_dir(variant: str, root: Path = cfg.DATA_PROCESSED) -> Path:
    return root / variant


def fit_fold(fold: int, variant: str, root: Path = cfg.DATA_PROCESSED):
    """Ajusta o Preprocessor no treino do fold e transforma os 3 conjuntos.

    Retorna (preprocessor, {"treino": X, "validacao": X, "teste": X}).
    """
    ids = load_ids(root)
    X_tr, _ = load_partition(fold, "treino", root)
    ids_tr = ids[(ids["fold"] == fold) & (ids["conjunto"] == "treino")].reset_index(drop=True)
    assert len(ids_tr) == len(X_tr), "ids_origem fora de sincronia com X_treino"
    pre = Preprocessor(**cfg.PREPROC_VARIANTS[variant]).fit(X_tr, original_mask=~ids_tr["repetido"].to_numpy())
    out = {"treino": pre.transform(X_tr)}
    for name in ("validacao", "teste"):
        X, _ = load_partition(fold, name, root)
        out[name] = pre.transform(X)
    return pre, out


def run_all(root: Path = cfg.DATA_PROCESSED, variants=None) -> pd.DataFrame:
    """Gera todas as variantes/folds, salva em data/processed/<variante>/ e devolve um resumo."""
    rows = []
    for variant in variants or cfg.PREPROC_VARIANTS:
        out_dir = variant_dir(variant, root)
        out_dir.mkdir(parents=True, exist_ok=True)
        for fold in range(1, cfg.K_FOLDS + 1):
            pre, sets = fit_fold(fold, variant, root)
            for name, Xp in sets.items():
                Xp.to_csv(out_dir / f"X_{name}_fold{fold}.csv", index=False)
                X_raw, _ = load_partition(fold, name, root)
                rows.append({"variante": variant, "fold": fold, "conjunto": name, "linhas": len(Xp),
                             "colunas": Xp.shape[1], "ausentes": int(Xp.isna().sum().sum()),
                             "categorias_desconhecidas": pre.unknown_categories(X_raw)})
            (out_dir / f"params_fold{fold}.json").write_text(
                json.dumps(pre.params(), indent=2, ensure_ascii=False), encoding="utf-8")
    return pd.DataFrame(rows)


def load_preprocessed(fold: int, name: str, variant: str = "neural", root: Path = cfg.DATA_PROCESSED):
    """Carrega (X, y) já pré-processados. y é o mesmo do particionamento (0/1)."""
    X = pd.read_csv(variant_dir(variant, root) / f"X_{name}_fold{fold}.csv")
    _, y = load_partition(fold, name, root)
    return X, y


# Verificações
def verify_preprocessing(root: Path = cfg.DATA_PROCESSED, variants=None) -> pd.DataFrame:
    """Levanta AssertionError se algo estiver errado; devolve a fração fora de [0,1] em val/teste."""
    from .data_loader import load_raw

    raw = load_raw()
    ids = load_ids(root)
    rows = []
    for variant in variants or cfg.PREPROC_VARIANTS:
        spec = cfg.PREPROC_VARIANTS[variant]
        for fold in range(1, cfg.K_FOLDS + 1):
            pre, sets = fit_fold(fold, variant, root)
            cols = list(sets["treino"].columns)
            for name, Xp in sets.items():
                X_raw, _ = load_partition(fold, name, root)
                assert len(Xp) == len(X_raw), f"{variant}/{fold}/{name}: nº de linhas mudou"
                assert list(Xp.columns) == cols, f"{variant}/{fold}/{name}: colunas diferentes do treino"
                assert not Xp.isna().any().any(), f"{variant}/{fold}/{name}: há ausentes"
                saved = pd.read_csv(variant_dir(variant, root) / f"X_{name}_fold{fold}.csv")
                assert saved.shape == Xp.shape, f"{variant}/{fold}/{name}: arquivo salvo difere do recalculado"

            # parâmetros vêm exclusivamente das linhas ORIGINAIS do treino (recalculado de forma independente)
            tr_ids = ids[(ids["fold"] == fold) & (ids["conjunto"] == "treino") & (~ids["repetido"])]["id_origem"]
            tr = raw.loc[tr_ids].copy()
            tr.loc[tr["TotalCharges"].isna() & (tr["tenure"] == 0), "TotalCharges"] = 0.0
            if spec["scale"]:
                for c in cfg.NUMERIC_COLS:
                    assert np.isclose(pre.params()["minmax"][c]["min"], tr[c].min()), f"min de {c} não vem só do treino"
                    assert np.isclose(pre.params()["minmax"][c]["max"], tr[c].max()), f"max de {c} não vem só do treino"
                orig_mask = ~ids[(ids["fold"] == fold) & (ids["conjunto"] == "treino")]["repetido"].to_numpy()
                tr_scaled = sets["treino"].loc[orig_mask, cfg.NUMERIC_COLS]
                assert np.allclose(tr_scaled.min(), 0) and np.allclose(tr_scaled.max(), 1), "treino fora de [0,1]"
            if spec["onehot"]:
                for name, Xp in sets.items():
                    oh = Xp.drop(columns=cfg.NUMERIC_COLS)
                    assert set(np.unique(oh.to_numpy())) <= {0.0, 1.0}, "one-hot não binário"
            for name in ("validacao", "teste"):
                X_raw, _ = load_partition(fold, name, root)
                assert pre.unknown_categories(X_raw) == 0, f"categoria nova em {name}"
                if spec["scale"]:
                    v = sets[name][cfg.NUMERIC_COLS]
                    rows.append({"variante": variant, "fold": fold, "conjunto": name,
                                 "pct_fora_0_1": round(100 * ((v < 0) | (v > 1)).any(axis=1).mean(), 3)})
    return pd.DataFrame(rows)


def main() -> None:
    cfg.ensure_dirs()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.FileHandler(cfg.LOGS / "preprocessamento.log", mode="w", encoding="utf-8"),
                                  logging.StreamHandler()])
    resumo = run_all()
    resumo.to_csv(cfg.REPORTS_TABLES / "preprocessamento_resumo.csv", index=False)
    fora = verify_preprocessing()
    log.info("Verificações OK.\n%s", resumo.groupby(["variante", "conjunto"])[["colunas"]].first().to_string())
    log.info("Fração de linhas com algum valor numérico fora de [0,1] (esperado: pequena, nada é recortado):\n%s",
             fora.to_string(index=False))


if __name__ == "__main__":
    main()
