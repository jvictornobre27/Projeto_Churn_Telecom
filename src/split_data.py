"""Particionamento em 3 folds (treino/validação/teste).

Esquema aplicado a cada fold:
  1. Separa-se por classe e divide-se cada classe em K partes estratificadas.
  2. Treino = parte `i`, validação = parte `i+1`, teste = parte `i+2` (rotação cíclica).
  3. Oversampling (repetição aleatória da classe minoritária) SÓ em treino e
     validação, até igualar a classe majoritária (50%/50%).
  4. Teste fica com as linhas originais, sem repetição, na proporção real.
  5. Cada conjunto é embaralhado ("aleatorização").

Nenhuma estatística é calculada aqui: normalização, imputação e codificação
são ajustadas depois, apenas no treino de cada fold.

Uso pela linha de comando (a partir da raiz do repositório):
    python -m src.split_data
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config as cfg
from .data_loader import load_raw

log = logging.getLogger("split_data")


# Construção
def make_folds(df: pd.DataFrame, k: int = cfg.K_FOLDS, seed: int = cfg.SEED) -> list[np.ndarray]:
    """Divide `df` em `k` folds disjuntos e estratificados pelo alvo.

    Retorna uma lista de `k` arrays com os `id_origem` de cada fold.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    ids = df.index.to_numpy()
    return [ids[test_idx] for _, test_idx in skf.split(df, df[cfg.TARGET])]


def oversample_minority(part: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Repete aleatoriamente (com reposição) a classe minoritária até igualar a majoritária.

    Todas as linhas originais são mantidas; apenas as cópias recebem
    `repetido=True`. Equivale ao RandomOverSampler com sampling_strategy=1.0,
    sem depender do imbalanced-learn.
    """
    counts = part[cfg.TARGET].value_counts()
    minority = counts.idxmin()
    n_extra = int(counts.max() - counts.min())
    rng = np.random.default_rng(seed)
    pool = part[part[cfg.TARGET] == minority]
    extra = pool.iloc[rng.choice(len(pool), size=n_extra, replace=True)].copy()
    extra["repetido"] = True
    return pd.concat([part, extra], ignore_index=True)


def _shuffle(part: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return part.iloc[rng.permutation(len(part))].reset_index(drop=True)


def build_partitions(
    df: pd.DataFrame,
    k: int = cfg.K_FOLDS,
    seed: int = cfg.SEED,
    oversample_train: bool = cfg.OVERSAMPLE_TRAIN,
    oversample_val: bool = cfg.OVERSAMPLE_VAL,
) -> dict[int, dict[str, pd.DataFrame]]:
    """Gera {fold: {"treino": df, "validacao": df, "teste": df}} (fold 1-based).

    Cada DataFrame tem as colunas originais + `id_origem` + `repetido`.
    """
    folds = make_folds(df, k, seed)
    oversample = {"treino": oversample_train, "validacao": oversample_val, "teste": False}
    parts: dict[int, dict[str, pd.DataFrame]] = {}
    for fold in range(1, k + 1):
        parts[fold] = {}
        for j, (name, fold_no) in enumerate(cfg.roles_for_fold(fold, k).items()):
            sub = df.loc[folds[fold_no - 1]].reset_index()   # traz id_origem para coluna
            sub["repetido"] = False
            sub_seed = seed + 100 * fold + j
            if oversample[name]:
                sub = oversample_minority(sub, sub_seed)
            parts[fold][name] = _shuffle(sub, sub_seed)
    return parts


# Verificações sobre independência, sem sobreposição, sem vazamento
def verify_partitions(
    parts: dict[int, dict[str, pd.DataFrame]],
    df: pd.DataFrame,
    oversample_train: bool = cfg.OVERSAMPLE_TRAIN,
    oversample_val: bool = cfg.OVERSAMPLE_VAL,
    tol_prop: float = 0.01,
) -> pd.DataFrame:
    """Valida o particionamento. Levanta AssertionError se algo estiver errado.

    Retorna uma tabela-resumo (fold x conjunto).
    """
    all_ids = set(df.index)
    global_rate = (df[cfg.TARGET] == cfg.POSITIVE_CLASS).mean()
    rows, test_ids_all = [], []

    for fold, sets in parts.items():
        uniq = {name: set(p[cfg.ID_COL]) for name, p in sets.items()}

        # 1) Sem sobreposição entre conjuntos
        for a, b in (("treino", "validacao"), ("treino", "teste"), ("validacao", "teste")):
            inter = uniq[a] & uniq[b]
            assert not inter, f"Fold {fold}: {len(inter)} registros em {a} e {b} ao mesmo tempo"

        # 2) Cobertura total: cada registro original em exatamente um conjunto
        assert set().union(*uniq.values()) == all_ids, f"Fold {fold}: registros faltando/sobrando"

        # 3) Teste: sem oversampling e proporção original
        t = sets["teste"]
        assert not t["repetido"].any(), f"Fold {fold}: teste tem linhas repetidas"
        assert t[cfg.ID_COL].is_unique, f"Fold {fold}: teste com id duplicado"
        rate_t = (t[cfg.TARGET] == cfg.POSITIVE_CLASS).mean()
        assert abs(rate_t - global_rate) <= tol_prop, f"Fold {fold}: teste fora da proporção original"
        test_ids_all.append(uniq["teste"])

        # 4) Treino/validação balanceados 50/50 quando há oversampling
        for name, flag in (("treino", oversample_train), ("validacao", oversample_val)):
            vc = sets[name][cfg.TARGET].value_counts()
            if flag:
                assert vc.nunique() == 1, f"Fold {fold}: {name} não ficou balanceado: {vc.to_dict()}"
            assert sets[name].loc[~sets[name]["repetido"], cfg.ID_COL].is_unique

        # 5) Os valores das features não foram alterados
        for name, p in sets.items():
            orig = df.loc[p[cfg.ID_COL], cfg.FEATURE_COLS + [cfg.TARGET]].reset_index(drop=True)
            assert orig.equals(p[cfg.FEATURE_COLS + [cfg.TARGET]].reset_index(drop=True)), \
                f"Fold {fold}: valores de {name} diferem do dataset original"

        for name, p in sets.items():
            n_pos = int((p[cfg.TARGET] == cfg.POSITIVE_CLASS).sum())
            rows.append({
                "fold": fold, "conjunto": name, "linhas": len(p),
                "registros_unicos": p[cfg.ID_COL].nunique(), "linhas_repetidas": int(p["repetido"].sum()),
                "churn_Yes": n_pos, "churn_No": len(p) - n_pos,
                "pct_churn": round(100 * n_pos / len(p), 2),
            })

    # 6) Entre folds, os testes são disjuntos e cobrem o dataset inteiro
    assert sum(len(s) for s in test_ids_all) == len(all_ids) and set().union(*test_ids_all) == all_ids, \
        "Os conjuntos de teste dos folds não particionam o dataset"

    return pd.DataFrame(rows)


def count_cross_set_duplicates(parts: dict[int, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Informativo: linhas com valores idênticos (features + alvo) em conjuntos diferentes.

    O dataset não tem ID de cliente e contém linhas idênticas por natureza; isso
    NÃO é sobreposição de registros, mas vale reportar.
    """
    rows = []
    cols = cfg.FEATURE_COLS + [cfg.TARGET]
    for fold, s in parts.items():
        key = {n: set(map(tuple, p.loc[~p["repetido"], cols].astype(str).to_numpy())) for n, p in s.items()}
        rows.append({
            "fold": fold,
            "treino_x_teste": len(key["treino"] & key["teste"]),
            "validacao_x_teste": len(key["validacao"] & key["teste"]),
            "treino_x_validacao": len(key["treino"] & key["validacao"]),
        })
    return pd.DataFrame(rows)


# Salvar / carregar
def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def save_partitions(
    parts: dict[int, dict[str, pd.DataFrame]],
    out_dir: Path = cfg.DATA_PROCESSED,
    seed: int = cfg.SEED,
) -> None:
    """Grava X_*/y_* por fold e conjunto, `ids_origem.csv` e `manifesto.json`.

    As linhas de `ids_origem.csv` seguem a mesma ordem dos arquivos X_/y_.
    Os X_ mantêm as features BRUTAS (sem imputar/codificar/normalizar).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ids_rows, shapes = [], {}
    for fold, sets in parts.items():
        for name, p in sets.items():
            tag = f"{name}_fold{fold}"
            p[cfg.FEATURE_COLS].to_csv(out_dir / f"X_{tag}.csv", index=False)
            y = (p[cfg.TARGET] == cfg.POSITIVE_CLASS).astype(int).rename(cfg.TARGET)
            y.to_csv(out_dir / f"y_{tag}.csv", index=False)
            ids_rows.append(p[[cfg.ID_COL, "repetido"]].assign(fold=fold, conjunto=name))
            shapes[tag] = list(p[cfg.FEATURE_COLS].shape)
    pd.concat(ids_rows)[["fold", "conjunto", cfg.ID_COL, "repetido"]].to_csv(out_dir / "ids_origem.csv", index=False)

    manifest = {
        "seed": seed,
        "k_folds": len(parts),
        "rotacao": {f: cfg.roles_for_fold(f, len(parts)) for f in parts},
        "oversampling": {"treino": cfg.OVERSAMPLE_TRAIN, "validacao": cfg.OVERSAMPLE_VAL, "teste": False},
        "alvo": {"coluna": cfg.TARGET, "positivo(1)": cfg.POSITIVE_CLASS},
        "csv_original_md5": _md5(cfg.RAW_FILE),
        "formas_X": shapes,
        "versoes": {"python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__},
    }
    (out_dir / "manifesto.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def load_partition(fold: int, name: str, out_dir: Path = cfg.DATA_PROCESSED) -> tuple[pd.DataFrame, pd.Series]:
    """Carrega (X, y) de um conjunto salvo. `name` in {"treino","validacao","teste"}."""
    tag = f"{name}_fold{fold}"
    X = pd.read_csv(out_dir / f"X_{tag}.csv")
    y = pd.read_csv(out_dir / f"y_{tag}.csv")[cfg.TARGET]
    return X, y


def load_ids(out_dir: Path = cfg.DATA_PROCESSED) -> pd.DataFrame:
    return pd.read_csv(out_dir / "ids_origem.csv")


# Execução ponta a ponta
def main() -> None:
    cfg.ensure_dirs()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(message)s",
        handlers=[logging.FileHandler(cfg.LOGS / "particionamento.log", mode="w", encoding="utf-8"),
                  logging.StreamHandler()],
    )
    df = load_raw()
    log.info("Dataset: %d linhas | seed=%d | k=%d", len(df), cfg.SEED, cfg.K_FOLDS)
    parts = build_partitions(df)
    resumo = verify_partitions(parts, df)
    log.info("Verificações OK.\n%s", resumo.to_string(index=False))
    resumo.to_csv(cfg.REPORTS_TABLES / "particionamento_resumo.csv", index=False)
    save_partitions(parts)
    log.info("Partições salvas em %s", cfg.DATA_PROCESSED)


if __name__ == "__main__":
    main()
