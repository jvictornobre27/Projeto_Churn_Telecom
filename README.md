# Projeto Churn Telecom (IF702 – Redes Neurais)

## Integrantes
* João Victor Nobre de Medeiros Santos
* José Clebson de Souza Oliveira
* Leandro Luiz de Lima Freitas 
* Tomás Nascimento Santos

Previsão de churn com redes neurais e modelos de comparação. Etapas feitas: **EDA, particionamento em 3 folds e pré-processamento**.

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
│   └── 04_preprocessamento.ipynb         # imputação, faixas, one-hot, Min-Max (ajustados só no treino)
├── src/
│   ├── config.py       # caminhos, semente, colunas, rotação dos folds
│   ├── data_loader.py  # leitura do CSV original
│   ├── split_data.py   # particionamento, verificações, salvar/carregar
│   └── preprocess.py   # pré-processamento por fold sem vazamento, verificações
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