"""Leitura do dataset original (data/raw), que nunca é modificado."""
from pathlib import Path

import pandas as pd

from .config import FEATURE_COLS, ID_COL, NEGATIVE_CLASS, POSITIVE_CLASS, RAW_FILE, TARGET


def load_raw(path: Path = RAW_FILE, validate: bool = True) -> pd.DataFrame:
    """Lê o CSV original.

    O índice do DataFrame (`id_origem`) é a posição da linha no CSV. Como o
    dataset não tem identificador de cliente, é ele que permite rastrear cada
    registro depois do particionamento/oversampling.
    """
    df = pd.read_csv(path)
    df.index.name = ID_COL
    if validate:
        _validate(df)
    return df


def _validate(df: pd.DataFrame) -> None:
    missing = set(FEATURE_COLS + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no dataset: {sorted(missing)}")
    extra = set(df.columns) - set(FEATURE_COLS + [TARGET])
    if extra:
        raise ValueError(f"Colunas inesperadas no dataset: {sorted(extra)}")
    labels = set(df[TARGET].dropna().unique())
    if labels != {NEGATIVE_CLASS, POSITIVE_CLASS}:
        raise ValueError(f"Valores inesperados no alvo {TARGET}: {labels}")
    if df[TARGET].isna().any():
        raise ValueError("Há valores ausentes no alvo.")
