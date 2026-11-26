"""KAN de referência com splines lineares em cada conexão."""
import torch
from torch import nn


class CamadaKAN(nn.Module):
    def __init__(self, entradas, saidas, minimo, maximo, intervalos):
        super().__init__()
        # Cada nó interior permite uma mudança de inclinação da spline.
        nos = torch.linspace(minimo, maximo, intervalos + 1)[1:-1]
        self.register_buffer("nos", nos)
        self.projecao = nn.Linear(entradas * intervalos, saidas)

    def forward(self, x):
        # A base linear e as dobradiças formam uma spline por entrada e saída.
        base = torch.cat([x.unsqueeze(-1), torch.relu(x.unsqueeze(-1) - self.nos)], dim=-1)
        return self.projecao(base.flatten(1))


def criar_spline(entradas, config):
    return nn.Sequential(CamadaKAN(entradas, config["largura"], 0, 1, config["intervalos"]),
                           CamadaKAN(config["largura"], 1, -1, 1, config["intervalos"]),
                           nn.Flatten(start_dim=0))


def ajustar(dados, config, semente, criar_modelo=criar_spline, *, x_teste=None):
    from copy import deepcopy
    from time import perf_counter

    inicio = perf_counter()
    torch.manual_seed(semente)
    tensores = {}
    for conjunto, (X, y) in dados.items():
        tensores[conjunto] = (torch.tensor(X.to_numpy(), dtype=torch.float32),
                              torch.tensor(y.to_numpy(), dtype=torch.float32).reshape(-1))
    x, y = tensores["treino"]
    modelo = criar_modelo(x.shape[1], config)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=config["lr"])
    perda = nn.BCEWithLogitsLoss()
    historico = []
    melhor = float("inf")
    for epoca in range(1, config["max_epocas"] + 1):
        for indices in torch.randperm(len(y)).split(config["lote"]):
            otimizador.zero_grad()
            perda(modelo(x[indices]), y[indices]).backward()
            otimizador.step()
        # Avalia os conjuntos completos com os mesmos pesos ao fim da época.
        with torch.no_grad():
            perdas = {nome: perda(modelo(a), b).item() for nome, (a, b) in tensores.items()}
        historico.append(perdas)
        if perdas["validacao"] < melhor - config["min_delta"]:
            melhor = perdas["validacao"]
            pesos = deepcopy(modelo.state_dict())
            melhor_epoca = epoca
        if epoca - melhor_epoca >= config["paciencia"]:
            break

    # Restaura a época selecionada pela perda de validação.
    modelo.load_state_dict(pesos)
    with torch.no_grad():
        scores = {nome: (b.numpy(), modelo(a).sigmoid().numpy()) for nome, (a, b) in tensores.items()}
        # O teste é pontuado somente após restaurar o checkpoint de validação.
        if x_teste is not None:
            entrada = torch.tensor(x_teste.to_numpy(), dtype=torch.float32)
            scores["prob_teste"] = modelo(entrada).sigmoid().numpy()
    return scores, historico, melhor_epoca, perf_counter() - inicio
