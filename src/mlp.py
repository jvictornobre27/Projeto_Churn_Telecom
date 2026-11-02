"""Motor do MLP (PyTorch): rede, treino com parada antecipada, previsão e busca com Optuna.

O notebook `05_mlp.ipynb` só CHAMA este módulo; a lógica fica aqui para ser testada e reutilizada.

Decisões de projeto:
- Saída: 1 logit; a probabilidade de churn é sigmoid(logit) (é o que `src.metrics` espera).
- Perda é HIPERPARÂMETRO: "cross_entropy" (BCE sobre o logit) ou "mse" (erro quadrático sobre a
  probabilidade). As duas são medidas em todos os conjuntos por `src.metrics` (colunas `log_loss` e `mse`).
- Parada antecipada: perda de VALIDAÇÃO (a mesma perda usada no treino), paciência 20, máx. 10.000
  épocas (slides); ao parar, restaura os pesos da melhor época.
- Seleção de configuração: KS de validação (desempate: AUROC de validação). O TESTE nunca é
  carregado na busca nem no reranqueamento: só em `avaliar_configuracao(..., incluir_teste=True)`.
- Sem validação cruzada interna: o treino tem linhas repetidas (oversampling) e dividir em folds
  internos vazaria cópias entre treino e validação. Usamos a validação fixa do fold.
- Reprodutibilidade: `fit(..., seed=...)` fixa Python/NumPy/PyTorch e a ordem dos mini-lotes.
  Padrão em CPU (rede pequena, e determinístico); em GPU pode haver pequenas diferenças.
"""
from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn

from . import config as cfg
from .metrics import evaluate_sets, ks_score
from .preprocess import load_preprocessed

ACTIVATIONS = ("logistic", "tanh", "relu")
OPTIMIZERS = ("adam", "rmsprop", "sgd", "adadelta")
LOSSES = ("cross_entropy", "mse")
# Faixa da taxa de aprendizagem por otimizador (Adadelta usa escala própria, padrão lr=1.0)
LR_RANGE = {"adam": (1e-4, 1e-2), "rmsprop": (1e-4, 1e-2), "sgd": (1e-3, 1e-1), "adadelta": (0.1, 2.0)}
MAX_EPOCAS = 10_000
PACIENCIA = 20
# Sem um limite mínimo, otimizadores lentos (SGD/Adadelta com taxa baixa) melhoram ~1e-5 por época,
# zeram a paciência para sempre e rodam até as 10.000 épocas. Como no `min_delta` do Keras, só
# conta como melhora uma queda de pelo menos MIN_DELTA na perda de validação.
MIN_DELTA = 1e-4
OPTUNA_DIR = cfg.ROOT / "models" / "optuna"   # `models/` não vai para o Git


# Configuração da rede (hiperparâmetros)
@dataclass
class MLPConfig:
    """Todos os hiperparâmetros de uma rede. O padrão é a rede pequena inicial do slide (1 camada, 10 unidades)."""
    neuronios: tuple = (10,)          # 1 ou 2 camadas ocultas
    ativacao: str = "relu"            # logistic | tanh | relu
    dropout: float = 0.0
    perda: str = "cross_entropy"      # cross_entropy | mse
    otimizador: str = "adam"          # adam | rmsprop | sgd | adadelta
    lr: float = 1e-3
    weight_decay: float = 0.0         # regularização L2
    momentum: float = 0.0             # só usado pelo SGD
    batch_size: int = 64
    max_epocas: int = MAX_EPOCAS
    paciencia: int = PACIENCIA
    min_delta: float = MIN_DELTA      # melhora mínima da perda de validação para contar como progresso

    def __post_init__(self):
        self.neuronios = tuple(int(n) for n in self.neuronios)
        if not 1 <= len(self.neuronios) <= 2:
            raise ValueError("neuronios deve ter 1 ou 2 camadas ocultas")
        for campo, validos in (("ativacao", ACTIVATIONS), ("otimizador", OPTIMIZERS), ("perda", LOSSES)):
            if getattr(self, campo) not in validos:
                raise ValueError(f"{campo}={getattr(self, campo)!r} inválido; use {validos}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout deve estar em [0, 1)")


def config_to_dict(c: MLPConfig) -> dict:
    d = asdict(c)
    d["neuronios"] = list(d["neuronios"])
    return d


def config_from_dict(d: dict) -> MLPConfig:
    return MLPConfig(**d)


def save_config(c: MLPConfig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_to_dict(c), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_config(path: Path) -> MLPConfig:
    return config_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# Rede inicial (E0): pequena, (1 camada, 10 unidades)
BASELINE = MLPConfig()


# Rede
_ACT = {"logistic": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU}


class MLP(nn.Module):
    """Perceptron multicamadas com 1 logit de saída."""

    def __init__(self, n_in: int, neuronios: tuple, ativacao: str = "relu", dropout: float = 0.0):
        super().__init__()
        layers, d = [], n_in
        for h in neuronios:
            layers += [nn.Linear(d, h), _ACT[ativacao]()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)   # logit; probabilidade = sigmoid(logit)


def _loss_fn(nome: str):
    if nome == "cross_entropy":
        return lambda logit, y: F.binary_cross_entropy_with_logits(logit, y)
    return lambda logit, y: F.mse_loss(torch.sigmoid(logit), y)


def _make_optimizer(params, c: MLPConfig):
    if c.otimizador == "adam":
        return torch.optim.Adam(params, lr=c.lr, weight_decay=c.weight_decay)
    if c.otimizador == "rmsprop":
        return torch.optim.RMSprop(params, lr=c.lr, weight_decay=c.weight_decay)
    if c.otimizador == "sgd":
        return torch.optim.SGD(params, lr=c.lr, momentum=c.momentum, weight_decay=c.weight_decay)
    return torch.optim.Adadelta(params, lr=c.lr, weight_decay=c.weight_decay)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tensor(a, device, dtype=torch.float32):
    arr = np.asarray(a, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("entrada com NaN/inf")
    return torch.as_tensor(arr, dtype=dtype, device=device)


# Treino da rede
@dataclass
class FitResult:
    model: nn.Module
    history: pd.DataFrame      # por época: treino_loss, val_loss, val_ks
    best_epoch: int
    best_val_loss: float
    n_epocas: int              # épocas efetivamente rodadas
    parou_cedo: bool           # True se parou por paciência (False = atingiu max_epocas)


def fit(X_tr, y_tr, X_va, y_va, config: MLPConfig = BASELINE, seed: int = cfg.SEED,
        device: str = "cpu", verbose: bool = False, registrar_ks: bool = True,
        threads: int | None = 1) -> FitResult:
    """Treina com mini-lotes e parada antecipada na perda de validação. Devolve a rede da MELHOR época.

    `registrar_ks=False` pula o KS de validação por época (mais rápido; usado na busca).
    `threads=1` (padrão) é mais rápido para redes tão pequenas; `None` mantém a configuração do PyTorch.
    """
    anterior = torch.get_num_threads()
    if threads:
        torch.set_num_threads(threads)
    try:
        return _fit(X_tr, y_tr, X_va, y_va, config, seed, device, verbose, registrar_ks)
    finally:
        torch.set_num_threads(anterior)


def _fit(X_tr, y_tr, X_va, y_va, config, seed, device, verbose, registrar_ks) -> FitResult:
    set_seed(seed)
    Xt, yt = _tensor(X_tr, device), _tensor(y_tr, device)
    Xv, yv = _tensor(X_va, device), _tensor(y_va, device)
    y_va_np = np.asarray(y_va).ravel()
    model = MLP(Xt.shape[1], config.neuronios, config.ativacao, config.dropout).to(device)
    opt = _make_optimizer(model.parameters(), config)
    loss_fn = _loss_fn(config.perda)
    gen = torch.Generator().manual_seed(seed)       # ordem dos mini-lotes, reprodutível

    n = len(Xt)
    best, best_epoch, best_state, ruins, rows = float("inf"), 0, None, 0, []
    parou_cedo = False
    for epoca in range(1, config.max_epocas + 1):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        soma = torch.zeros((), device=device)
        for i in range(0, n, config.batch_size):
            idx = perm[i:i + config.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            soma += loss.detach() * len(idx)
        model.eval()
        with torch.no_grad():
            logit_v = model(Xv)
            val_loss = float(loss_fn(logit_v, yv))
            p_v = torch.sigmoid(logit_v).cpu().numpy()
        if not np.isfinite(val_loss) or not np.isfinite(p_v).all():
            break                                   # divergiu (ex.: taxa de aprendizagem alta demais)
        rows.append({"epoca": epoca, "treino_loss": float(soma) / n, "val_loss": val_loss,
                     "val_ks": ks_score(y_va_np, p_v) if registrar_ks else float("nan")})
        if verbose and epoca % 10 == 0:
            print(f"época {epoca:4d} | treino {rows[-1]['treino_loss']:.4f} | val {val_loss:.4f} | KS val {rows[-1]['val_ks']:.4f}")
        if val_loss < best - config.min_delta:
            best, best_epoch, ruins = val_loss, epoca, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            ruins += 1
            if ruins >= config.paciencia:
                parou_cedo = True
                break
    if best_state is None:
        raise FloatingPointError("o treino divergiu antes de qualquer época válida")
    model.load_state_dict(best_state)
    model.eval()
    return FitResult(model, pd.DataFrame(rows), best_epoch, best, len(rows), parou_cedo)


def predict_proba(model: nn.Module, X, device: str = "cpu") -> np.ndarray:
    """Probabilidade de churn (vetor 1-D em [0, 1]), pronta para `src.metrics`."""
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(_tensor(X, device))).cpu().numpy().astype(float)


# Dados e avaliação
def load_xy(fold: int, variante: str, conjuntos=("treino", "validacao")) -> dict:
    """{conjunto: (X, y)} pré-processado. Por padrão NÃO carrega o teste."""
    return {c: load_preprocessed(fold, c, variante) for c in conjuntos}


def avaliar_configuracao(config: MLPConfig, variante: str, fold: int, seeds=(cfg.SEED,),
                         etapa: str = "E0_baseline", obs: str = "", incluir_teste: bool = True,
                         device: str = "cpu", modelo: str = "mlp"):
    """Treina a configuração em cada semente e devolve (DataFrame de resultados, saídas por semente).

    O DataFrame segue o formato de `src.metrics` (uma linha por semente e conjunto), pronto para
    `save_results`. `saidas[semente] = {"fit": FitResult, "scores": {...}}` serve para gráficos.
    O teste só é carregado com `incluir_teste=True` (registro da escada; NUNCA para escolher).
    """
    data = load_xy(fold, variante, cfg.SET_NAMES if incluir_teste else ("treino", "validacao"))
    (X_tr, y_tr), (X_va, y_va) = data["treino"], data["validacao"]
    frames, saidas = [], {}
    for s in seeds:
        r = fit(X_tr, y_tr, X_va, y_va, config, seed=s, device=device)
        scores = {c: (y, predict_proba(r.model, X, device)) for c, (X, y) in data.items()}
        hp = {**config_to_dict(config), "melhor_epoca": r.best_epoch, "epocas_rodadas": r.n_epocas,
              "parou_cedo": r.parou_cedo}
        frames.append(evaluate_sets(modelo, variante, fold, scores, hiperparametros=hp, obs=obs,
                                    etapa=etapa, semente=s))
        saidas[s] = {"fit": r, "scores": scores}
    return pd.concat(frames, ignore_index=True), saidas


# Busca de hiperparâmetros com Optuna
def suggest_config(trial) -> MLPConfig:
    """Espaço de busca dos slides: camadas (1-2), neurônios, taxa de aprendizagem, ativação,
    otimizador, dropout, regularização L2 e a PERDA (cross-entropy vs MSE)."""
    n1 = trial.suggest_int("neuronios_1", 5, 128, log=True)
    neuronios = (n1,)
    if trial.suggest_int("camadas", 1, 2) == 2:
        neuronios = (n1, trial.suggest_int("neuronios_2", 5, 64, log=True))
    otim = trial.suggest_categorical("otimizador", list(OPTIMIZERS))
    return MLPConfig(
        neuronios=neuronios,
        ativacao=trial.suggest_categorical("ativacao", list(ACTIVATIONS)),
        dropout=round(trial.suggest_float("dropout", 0.0, 0.5, step=0.1), 1),
        perda=trial.suggest_categorical("perda", list(LOSSES)),
        otimizador=otim,
        lr=trial.suggest_float(f"lr_{otim}", *LR_RANGE[otim], log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        momentum=trial.suggest_categorical("momentum_sgd", [0.0, 0.9]) if otim == "sgd" else 0.0,
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
    )


def make_objective(variante: str, fold: int = 1, seed: int = cfg.SEED, device: str = "cpu"):
    """Função objetivo do Optuna: maximiza o KS de VALIDAÇÃO. Só treino e validação são lidos."""
    import optuna

    (X_tr, y_tr), (X_va, y_va) = load_xy(fold, variante).values()

    def objective(trial):
        c = suggest_config(trial)
        try:
            r = fit(X_tr, y_tr, X_va, y_va, c, seed=seed, device=device, registrar_ks=False)
        except FloatingPointError:
            raise optuna.TrialPruned("treino divergiu")
        p = predict_proba(r.model, X_va, device)
        ks = ks_score(y_va, p)
        trial.set_user_attr("val_auroc", float(roc_auc_score(y_va, p)))
        trial.set_user_attr("melhor_epoca", r.best_epoch)
        trial.set_user_attr("config", config_to_dict(c))
        return ks

    return objective


def run_study(variante: str, n_trials: int = 100, fold: int = 1, seed: int = cfg.SEED,
              device: str = "cpu", storage_dir: Path = OPTUNA_DIR, mostrar_progresso: bool = True,
              timeout: float | None = None):
    """Cria (ou RETOMA) o estudo da variante e roda até completar `n_trials` tentativas no total.

    O estudo fica em SQLite (`models/optuna/`, fora do Git); rodar de novo continua de onde parou.
    `timeout` (segundos) interrompe a busca; as tentativas feitas ficam salvas e podem ser retomadas.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    nome = f"mlp_{variante}_fold{fold}"
    storage = f"sqlite:///{(storage_dir / (nome + '.db')).as_posix()}"
    # A semente do sampler soma o nº de tentativas já feitas: ao retomar, o sorteio continua
    # de onde parou em vez de repetir as primeiras sugestões (reprodutível para o mesmo histórico).
    try:
        feitas = len(optuna.load_study(study_name=nome, storage=storage).trials)
    except KeyError:
        feitas = 0
    study = optuna.create_study(
        study_name=nome, storage=storage, load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed + feitas))
    faltam = n_trials - len(study.trials)
    if faltam > 0:
        study.optimize(make_objective(variante, fold, seed, device), n_trials=faltam,
                       timeout=timeout, show_progress_bar=mostrar_progresso)
    return study


def trials_table(study) -> pd.DataFrame:
    """Todas as tentativas concluídas (KS e AUROC de validação, época, configuração), da melhor para a pior."""
    rows = []
    for t in study.trials:
        if t.state.name != "COMPLETE":
            continue
        rows.append({"trial": t.number, "val_ks": t.value, "val_auroc": t.user_attrs["val_auroc"],
                     "melhor_epoca": t.user_attrs["melhor_epoca"], **t.user_attrs["config"]})
    df = pd.DataFrame(rows)
    return df.sort_values(["val_ks", "val_auroc"], ascending=False).reset_index(drop=True)


def reavaliar_candidatas(study, variante: str, k: int = 5, seeds=(42, 43, 44), fold: int = 1,
                         device: str = "cpu"):
    """Reavalia as k melhores tentativas com várias sementes (só validação) e escolhe a melhor.

    Uma tentativa isolada pode ser sorte de semente; a média das sementes é um critério mais robusto.
    Ordena por KS médio de validação (desempate: AUROC médio). Devolve (tabela, melhor MLPConfig).
    """
    top = trials_table(study).head(k)
    (X_tr, y_tr), (X_va, y_va) = load_xy(fold, variante).values()
    linhas, configs = [], {}
    for _, row in top.iterrows():
        c = config_from_dict({f: row[f] if f != "neuronios" else tuple(row[f]) for f in MLPConfig.__dataclass_fields__
                              if f in row})
        ks, auc = [], []
        for s in seeds:
            try:
                r = fit(X_tr, y_tr, X_va, y_va, c, seed=s, device=device, registrar_ks=False)
            except FloatingPointError:
                continue
            p = predict_proba(r.model, X_va, device)
            ks.append(ks_score(y_va, p))
            auc.append(roc_auc_score(y_va, p))
        if not ks:
            continue
        linhas.append({"trial": int(row["trial"]), "ks_trial_original": row["val_ks"],
                       "val_ks_media": float(np.mean(ks)), "val_ks_dp": float(np.std(ks, ddof=1)) if len(ks) > 1 else 0.0,
                       "val_auroc_media": float(np.mean(auc)), "sementes": len(ks)})
        configs[int(row["trial"])] = c
    if not linhas:
        raise RuntimeError("nenhuma candidata treinou com sucesso")
    tab = (pd.DataFrame(linhas).sort_values(["val_ks_media", "val_auroc_media"], ascending=False)
           .reset_index(drop=True))
    return tab, configs[int(tab.loc[0, "trial"])]


def etapa_busca(variante: str, etapa: str, n_trials: int = 100, fold: int = 1, k: int = 5,
                seeds_rerank=(42, 43, 44), seeds_registro=(42, 43, 44), device: str = "cpu",
                timeout: float | None = None, storage_dir: Path = OPTUNA_DIR,
                out_dir: Path = cfg.REPORTS_TABLES, mostrar_progresso: bool = True) -> dict:
    """Uma etapa de busca completa na variante, usando SÓ treino e validação para escolher:

    1. Optuna (`n_trials` no total; retoma se já houver tentativas) maximizando o KS de validação;
    2. reavalia as `k` melhores com várias sementes e escolhe a de maior KS médio (desempate AUROC);
    3. salva a configuração (`mlp_melhor_<variante>.json`) e todas as tentativas (CSV);
    4. registra a configuração escolhida na `etapa`, com `seeds_registro` (o teste entra só no registro).
    """
    from .metrics import save_results

    out_dir = Path(out_dir)
    study = run_study(variante, n_trials, fold, device=device, storage_dir=storage_dir,
                      mostrar_progresso=mostrar_progresso, timeout=timeout)
    export_trials(study, variante, fold, out_dir)
    ranking, melhor = reavaliar_candidatas(study, variante, k, seeds_rerank, fold, device)
    save_config(melhor, out_dir / f"mlp_melhor_{variante}.json")
    obs = f"melhor de {len(study.trials)} tentativas Optuna; escolha por KS de validação"
    res, saidas = avaliar_configuracao(melhor, variante, fold, seeds_registro, etapa=etapa, obs=obs,
                                       device=device)
    save_results(res, out_dir)
    return {"study": study, "ranking": ranking, "config": melhor, "resultados": res, "saidas": saidas}


def melhor_etapa(results: pd.DataFrame, etapas, fold: int = 1, modelo: str = "mlp"):
    """Entre as `etapas` dadas, a de maior KS MÉDIO de VALIDAÇÃO no fold (desempate: AUROC).

    Devolve (etapa, variante, tabela de comparação). Nunca olha o teste.
    """
    d = results[(results["modelo"] == modelo) & (results["fold"] == fold)
                & (results["conjunto"] == "validacao") & (results["etapa"].isin(list(etapas)))]
    if d.empty:
        raise ValueError(f"sem resultados de validação para {list(etapas)} no fold {fold}")
    tab = (d.groupby(["etapa", "variante"])[["ks", "auroc"]].mean().reset_index()
           .sort_values(["ks", "auroc"], ascending=False).reset_index(drop=True))
    return tab.loc[0, "etapa"], tab.loc[0, "variante"], tab


def etapa_final(config: MLPConfig, variante: str, etapa: str = "E4_final_3folds", folds=(1, 2, 3),
                seeds=(42, 43, 44), device: str = "cpu", out_dir: Path = cfg.REPORTS_TABLES):
    """Retreina a configuração escolhida em cada fold e semente e registra tudo (3 folds x sementes)."""
    from .metrics import save_results

    frames, saidas = [], {}
    for f in folds:
        res, s = avaliar_configuracao(config, variante, f, seeds, etapa=etapa,
                                      obs="configuração escolhida no Fold 1, retreinada neste fold", device=device)
        frames.append(res)
        saidas[f] = s
    df = pd.concat(frames, ignore_index=True)
    save_results(df, out_dir)
    return df, saidas


def export_trials(study, variante: str, fold: int = 1, out_dir: Path = cfg.REPORTS_TABLES) -> Path:
    """Salva as tentativas em CSV (versionado), já que o banco do Optuna fica fora do Git."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"optuna_mlp_{variante}_fold{fold}_trials.csv"
    trials = trials_table(study)
    trials["neuronios"] = trials["neuronios"].map(lambda x: "-".join(map(str, x)) if not isinstance(x, str) else x)
    trials.to_csv(path, index=False)
    return path


# Gráfico de histórico de treino de uma rede (perda de treino e validação, KS de validação)
def plot_history(history: pd.DataFrame, best_epoch: int | None = None, titulo: str = "Curva de aprendizagem", ax=None):
    """Perda de treino e de validação por época (+ KS de validação, eixo da direita)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4), layout="constrained")
    ax.plot(history["epoca"], history["treino_loss"], label="perda treino", color="tab:gray")
    ax.plot(history["epoca"], history["val_loss"], label="perda validação", color="tab:blue")
    if best_epoch:
        ax.axvline(best_epoch, color="k", ls="--", lw=1, label=f"melhor época ({best_epoch})")
    ax.set(title=titulo, xlabel="Época", ylabel="Perda")
    ax2 = ax.twinx()
    ax2.plot(history["epoca"], history["val_ks"], color="tab:red", alpha=0.6, label="KS validação")
    ax2.set_ylabel("KS validação")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    return ax.figure
