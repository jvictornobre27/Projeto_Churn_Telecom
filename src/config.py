"""Configurações centrais do projeto de churn.

Tudo que é "constante do experimento" (caminhos, semente, nomes de colunas,
regras de rotação dos folds) fica aqui, para que VS Code, Colab e todos os
integrantes do grupo usem exatamente os mesmos valores.
"""
from pathlib import Path

# Caminhos (relativos à raiz do repositório, funcionam em VS Code e Colab)
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS_FIGURES = ROOT / "reports" / "figures"
REPORTS_TABLES = ROOT / "reports" / "tables"
LOGS = ROOT / "logs"

RAW_FILE = DATA_RAW / "customer_churn_telecom_services.csv"

# Reprodutibilidade
SEED = 42          # NÃO mudar depois de começarem os experimentos
K_FOLDS = 3

# Dados
TARGET = "Churn"
POSITIVE_CLASS = "Yes"   # classe minoritária (churn)
NEGATIVE_CLASS = "No"    # classe majoritária
ID_COL = "id_origem"     # posição da linha no CSV original (o dataset não tem ID de cliente)

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

# Particionamento
#   treino    : classe majoritária (original) + minoritária repetida até igualar
#   validacao : idem
#   teste     : ambas as classes ORIGINAIS, sem oversampling
SET_NAMES = ("treino", "validacao", "teste")
OVERSAMPLE_TRAIN = True
OVERSAMPLE_VAL = True

# Pré-processamento (ajustado SEMPRE e somente no treino de cada fold)

# Variantes geradas (registrar desempenho de forma evolutiva, como pedem os slides):
#   neural        : one-hot + Min-Max, todas as variáveis originais
#   neural_faixas : neural + faixas de tenure e MonthlyCharges (variáveis agregadas)
#   arvore        : só imputação; categóricas permanecem como texto, sem escala
PREPROC_VARIANTS = {
    "neural":        {"onehot": True,  "scale": True,  "bins": False},
    "neural_faixas": {"onehot": True,  "scale": True,  "bins": True},
    "arvore":        {"onehot": False, "scale": False, "bins": False},
}
# Faixas de tenure (meses), escolhidas na EDA (03): (-inf,6], (6,12], (12,24], (24,48], (48,inf)
TENURE_BIN_EDGES = [6, 12, 24, 48]
TENURE_BIN_LABELS = ["0-6", "7-12", "13-24", "25-48", "49+"]
# MonthlyCharges: quartis calculados no treino (original, sem repetidas)
MONTHLY_BIN_QUANTILES = [0.25, 0.5, 0.75]
MONTHLY_BIN_LABELS = ["Q1", "Q2", "Q3", "Q4"]


def roles_for_fold(fold: int, k: int = K_FOLDS) -> dict:
    """Rotação cíclica dos folds. `fold` é 1-based.

    fold 1 -> treino=1, validacao=2, teste=3
    fold 2 -> treino=2, validacao=3, teste=1
    fold 3 -> treino=3, validacao=1, teste=2
    """
    return {
        "treino": fold,
        "validacao": fold % k + 1,
        "teste": (fold + 1) % k + 1,
    }


def ensure_dirs() -> None:
    """Cria as pastas que o código gera (idempotente)."""
    for d in (DATA_INTERIM, DATA_PROCESSED, REPORTS_FIGURES, REPORTS_TABLES, LOGS):
        d.mkdir(parents=True, exist_ok=True)
