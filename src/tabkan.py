"""Variante ChebyshevKAN implementada em PyTorch para o protocolo do projeto."""
import torch
from torch import nn


class CamadaChebyshev(nn.Module):
    def __init__(self, entradas, saidas, grau):
        super().__init__()
        self.grau = grau
        # Cada conexão aprende um coeficiente por polinômio, inclusive o constante.
        self.projecao = nn.Linear(entradas * (grau + 1), saidas, bias=False)

    def forward(self, x):
        # tanh leva cada entrada ao domínio dos polinômios de Chebyshev.
        z = x.tanh()
        termos = [torch.ones_like(z), z]
        for _ in range(2, self.grau + 1):
            termos.append(2 * z * termos[-1] - termos[-2])
        return self.projecao(torch.stack(termos, dim=-1).flatten(1))


def criar_chebyshev(entradas, config):
    largura, grau = config["largura"], config["grau"]
    return nn.Sequential(CamadaChebyshev(entradas, largura, grau),
                         CamadaChebyshev(largura, 1, grau),
                         nn.Flatten(start_dim=0))


class ChebyshevMixer(nn.Module):
    """Mixer compacto com um bloco residual nos eixos de atributos e canais."""

    def __init__(self, entradas, canais, largura, grau, normalizar_atributos=True):
        super().__init__()
        self.embedding = nn.Linear(1, canais)
        self.norm_atributos = nn.LayerNorm(canais) if normalizar_atributos else nn.Identity()
        self.norm_canais = nn.LayerNorm(canais)
        self.atributos = nn.Sequential(CamadaChebyshev(entradas, largura, grau),
                                       CamadaChebyshev(largura, entradas, grau))
        self.canais = nn.Sequential(CamadaChebyshev(canais, largura, grau),
                                    CamadaChebyshev(largura, canais, grau))
        self.saida = nn.Linear(canais, 1)

    def forward(self, x):
        h = self.embedding(x.unsqueeze(-1))
        lote, atributos, canais = h.shape
        # Cada canal combina os atributos do mesmo cliente.
        z = self.norm_atributos(h).transpose(1, 2).reshape(-1, atributos)
        h = h + self.atributos(z).reshape(lote, canais, atributos).transpose(1, 2)
        # Cada atributo combina seus canais antes da média por cliente.
        z = self.norm_canais(h).reshape(-1, canais)
        h = h + self.canais(z).reshape(lote, atributos, canais)
        return self.saida(h.mean(dim=1)).squeeze(-1)


def criar_mixer(entradas, config):
    return ChebyshevMixer(entradas, config["canais"], config["largura"], config["grau"],
                         normalizar_atributos=config.get("normalizar_atributos", True))
