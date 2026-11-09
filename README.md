# Projeto Churn Telecom (IF702 – Redes Neurais)

## Integrantes
* João Victor Nobre de Medeiros Santos
* José Clebson de Souza Oliveira
* Leandro Luiz de Lima Freitas 
* Tomás Nascimento Santos

Previsão de churn com redes neurais e modelos de comparação. Etapas feitas: **EDA, particionamento em 3 folds, pré-processamento, métricas compartilhadas, MLP (PyTorch), XGBoost e TabPFN v2**.

## Estrutura

```text
Projeto_Churn_Telecom/
├── data/
│   ├── raw/          # CSV original. NUNCA é alterado
│   ├── interim/      # intermediários (gerados por código)
│   └── processed/    # X_*/y_* por fold (brutos), ids_origem.csv, manifesto.json (versionados: todos usam os mesmos folds)
│       ├── neural/ neural_faixas/ arvore/   # pré-processados por fold + params_fold*.json
├── notebooks/
│   ├── 00_configuracao_ambiente.ipynb    # setup VS Code/Colab
│   ├── 01_analise_inicial_dados.ipynb    # visão estrutural do dataset
│   ├── 02_particionamento_dados.ipynb    # 3 folds, oversampling, verificações, salvar
│   ├── 03_eda_treino_fold1.ipynb         # EDA só no treino do Fold 1
│   ├── 04_preprocessamento.ipynb         # imputação, faixas, one-hot, Min-Max (ajustados só no treino)
│   ├── 05_mlp.ipynb                      # MLP: escada de evolução E0→E4 (Optuna, 3 folds × 3 sementes)
│   ├── 06_xgboost.ipynb                  # XGBoost: escada E0→E3 (Optuna, 3 folds × 3 sementes)
│   └── 07_tabpfn.ipynb                   # TabPFN v2: escada E0→E4 (sem busca; varia a entrada)
├── src/
│   ├── config.py       # caminhos, semente, colunas, rotação dos folds
│   ├── data_loader.py  # leitura do CSV original
│   ├── split_data.py   # particionamento, verificações, salvar/carregar
│   ├── preprocess.py   # pré-processamento por fold sem vazamento, verificações
│   ├── metrics.py      # KS, AUROC, MSE... e formato ÚNICO de resultados (todos os modelos)
│   ├── mlp.py          # motor do MLP: rede, treino, parada antecipada, Optuna
│   ├── xgb.py          # motor do XGBoost: categóricas nativas, parada antecipada, Optuna
│   └── tabpfn_model.py # motor do TabPFN v2: contexto, variantes de entrada, registro
├── reports/{figures,tables}/
└── logs/
```

Regra de ouro: **lógica em `src/`, notebooks só chamam e mostram**. Assim o mesmo código roda no VS Code, no Colab e no terminal.

## Como rodar

**VS Code / local**
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
nbstripout --install                                   # uma vez: Git ignora saídas de notebook
python -m src.split_data                               # (opcional) regenera os folds pelo terminal
```

**Colab**: abra `notebooks/00_configuracao_ambiente.ipynb`, troque `REPO_URL` e rode a primeira célula (clona, instala e entra na pasta). Depois use `git pull` para pegar o trabalho do grupo; para enviar, faça commit/push pelo próprio Colab (com token) ou baixe o notebook e commite pelo VS Code.

## Particionamento

- 3 folds estratificados pelo alvo; rotação: fold 1 → treino=1/val=2/teste=3; fold 2 → 2/3/1; fold 3 → 3/1/2.
- Oversampling (repetição aleatória da classe minoritária) **só em treino e validação** → 50%/50%.
- Teste: linhas originais, sem repetição, proporção real (~26,5% de churn).
- `ids_origem.csv` rastreia cada linha (`id_origem` = linha no CSV original; `repetido` marca as cópias).
- Os `X_*` guardam features **brutas**. Imputação, codificação e normalização devem ser **ajustadas no treino** de cada fold e aplicadas em validação/teste.
- `SEED = 42` (`src/config.py`): não mudar depois que os experimentos começarem.

## Pré-processamento (`python -m src.preprocess`)

Tudo que é aprendido (categorias, min/max, quartis das faixas) é ajustado **só nas linhas originais do treino** de cada fold e aplicado em validação/teste. Nada é removido nem recortado em validação/teste.
- `TotalCharges` ausente → 0 (todos têm `tenure == 0`). Sem remoção de duplicadas e sem capping (a EDA não achou outliers).
- Variantes: `neural` (one-hot + Min-Max), `neural_faixas` (+ faixas de `tenure`/`MonthlyCharges`), `arvore` (só imputação, categóricas como texto).
- Parâmetros auditáveis em `data/processed/<variante>/params_fold*.json`.

```python
from src.preprocess import load_preprocessed
X_tr, y_tr = load_preprocessed(1, "treino", "neural")
```

## Carregar os dados brutos particionados

```python
from src.split_data import load_partition
X_tr, y_tr = load_partition(1, "treino")
X_va, y_va = load_partition(1, "validacao")
X_te, y_te = load_partition(1, "teste")
```

## Plano

Hiperparâmetros só com o Fold 1 → melhores modelos retreinados nos Folds 2 e 3 → média e desvio padrão das métricas (KS principal, AUROC, F1, recall, precision, MSE).

## Registro de resultados e evolução para os modelos

Todo modelo salva em `reports/tables/resultados_<modelo>.csv` pelo mesmo formato (`src/metrics.py`). A chave de cada linha é
`modelo, etapa, variante, fold, semente, conjunto`:
- **`etapa`**: degrau da escada para registrar o desempenho de forma evolutiva, ex.: `E0_baseline`, `E1_faixas`, ...
  O primeiro modelo usa **todas as variáveis**; cada mudança vira uma etapa nova. O prefixo `E<n>` define a ordem nos gráficos.
- **`semente`**: semente do treino. `summarize_folds` faz a média das sementes dentro de cada fold e só então calcula média ± desvio entre folds.
- Escolhas (hiperparâmetros, variante, etapa) usam **só o KS de validação**; o teste é registrado, não usado para escolher.

```python
from src.metrics import evaluate_sets, save_results, load_results, summarize_folds, plot_evolution
res = evaluate_sets("xgboost", "arvore", fold=1, scores={...}, etapa="E0_baseline", semente=42)
save_results(res)
plot_evolution(load_results(), "xgboost", metric="ks", fold=1)
```

## MLP (`notebooks/05_mlp.ipynb`)

Escada: E0 (rede pequena, todas as variáveis) → E1 (+faixas) → E2/E3 (busca Optuna em `neural`/`neural_faixas`, 100 tentativas, perda cross-entropy vs MSE como hiperparâmetro)
→ E4 (melhor configuração em 3 folds × 3 sementes). A busca leva ~15–40 min por variante em CPU e **retoma** se for interrompida (estudos em `models/optuna/`, fora do Git).
Os CSV/JSON de `reports/tables/` são versionados.
## XGBoost (`notebooks/06_xgboost.ipynb`)

Escada: E0 (padrão da biblioteca, `arvore` com categóricas nativas) → E1 (Optuna em `arvore`) → E2 (Optuna em `neural`, one-hot)
→ E3 (melhor entre E1/E2 pelo KS de validação, 3 folds × 3 sementes). Parada antecipada pela log-loss de validação (paciência 20).
Roda em ~3 min em CPU.

## TabPFN v2 (`notebooks/07_tabpfn.ipynb`)

Transformer pré-treinado: o treino entra como contexto, sem gradiente e sem busca de hiperparâmetros. A escada varia a entrada:
E0 (`neural`) → E1 (`arvore`, categóricas nativas) → E2 (contexto sem as cópias do oversampling) → E3 (32 membros no ensemble)
→ E4 (melhor pelo KS de validação, 3 folds × 3 sementes).
- A versão é fixada em **v2** (`ModelVersion.V2`). Os pesos do v2 baixam do HuggingFace sem token; as versões 2.5+ (padrão do pacote) exigem licença.
- **Colab com GPU** (T4) recomendado. Em CPU funciona, mais devagar.
