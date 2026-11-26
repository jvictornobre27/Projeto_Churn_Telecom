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
