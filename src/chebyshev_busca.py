"""Busca de hiperparâmetros da ChebyshevKAN no treino e validação do fold 1."""
from pathlib import Path

import optuna
import pandas as pd
from optuna.trial import TrialState

from . import config as cfg
from .kan import ajustar
from .metrics import ks_score
from .tabkan import criar_chebyshev


def buscar_chebyshev(dados, referencia, raiz):
    # A base SQLite permite retomar o mesmo estudo e completar o orçamento.
    pasta = Path(raiz) / "models" / "optuna"
    pasta.mkdir(parents=True, exist_ok=True)
    banco = pasta / "chebyshev_neural_fold1.db"
    estudo = optuna.create_study(study_name="chebyshev_neural_fold1", direction="maximize",
                                 storage=f"sqlite:///{banco.as_posix()}", load_if_exists=True,
                                 sampler=optuna.samplers.TPESampler(seed=42))
    if not estudo.trials:
        estudo.enqueue_trial({k: referencia[k] for k in ("largura", "grau", "lr", "lote")})

    def objetivo(trial):
        parametros = dict(referencia, largura=trial.suggest_int("largura", 5, 64, log=True),
                          grau=trial.suggest_int("grau", 1, 5),
                          lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                          lote=trial.suggest_categorical("lote", [32, 64, 128, 256]))
        scores, historia, epoca, _ = ajustar(dados, parametros, cfg.SEED, criar_chebyshev)
        trial.set_user_attr("melhor_epoca", epoca)
        trial.set_user_attr("epocas_rodadas", len(historia))
        return ks_score(*scores["validacao"])

    completas = estudo.get_trials(states=(TrialState.COMPLETE,))
    if len(completas) < 100:
        estudo.optimize(objetivo, n_trials=100 - len(completas))
    # Em empate, a tentativa de menor número entra primeiro.
    tabela = pd.DataFrame([dict(trial=t.number, ks=t.value, **t.params, **t.user_attrs)
                           for t in estudo.get_trials(states=(TrialState.COMPLETE,))])
    tabela.to_csv(cfg.REPORTS_TABLES / "optuna_chebyshev_neural_fold1_trials.csv", index=False)
    return tabela.sort_values(["ks", "trial"], ascending=[False, True])
