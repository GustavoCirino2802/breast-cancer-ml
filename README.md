# Breast Cancer (UCI #14): treinamento e inferência

Classificação do atributo **Class** (`no-recurrence-events` / `recurrence-events`) da base
[Breast Cancer da UCI](https://archive.ics.uci.edu/dataset/14/breast+cancer), com 286 pacientes
e 9 atributos categóricos.

## Estrutura

```
├── dados/
│   ├── breast-cancer.csv        # base com cabeçalho ("?" = ausente)
│   └── breast-cancer.names      # descrição original da UCI
├── src/preprocessamento.py      # codificação, imputação, escala e validação (usado pelos dois módulos)
├── treinamento.py               # MÓDULO DE TREINAMENTO
├── inferencia.py                # MÓDULO DE INFERÊNCIA
├── exemplos/                    # pacientes de exemplo em JSON
├── modelos/modelo.joblib        # melhor modelo (gerado pelo treinamento)
└── resultados/                  # ranking dos modelos e metadados do melhor
```

## Instalação

```bash
pip install -r requirements.txt
```

## 1. Módulo de treinamento

```bash
python treinamento.py
```

Etapas executadas:

1. **Análise exploratória**: dimensões, valores ausentes (8 em `node-caps` e 1 em `breast-quad`), distribuição da classe (70,3% / 29,7%) e taxa de recorrência por atributo.
2. **Divisão treino/teste estratificada** (80/20, `random_state=42`).
3. **Pré-processamento em pipeline**, ajustado apenas no treino para evitar vazamento de dados:
   - `age`, `tumor-size`, `inv-nodes`: faixas convertidas para o ponto médio (ordinal);
   - `deg-malig`: numérico (1–3);
   - `node-caps`, `breast`, `irradiat`: binários 0/1;
   - `menopause`, `breast-quad`: one-hot;
   - ausentes imputados pela moda; atributos numéricos padronizados (`StandardScaler`).
4. **Comparação de 10 modelos** com `GridSearchCV` e validação cruzada estratificada de 10 folds × 3 repetições, com **acurácia** como métrica: baseline (classe majoritária), Naive Bayes, KNN, Regressão Logística, Árvore de Decisão, SVM, Random Forest, AdaBoost, Gradient Boosting e MLP.
5. **Seleção do modelo de melhor acurácia** na validação cruzada.
6. **Avaliação no conjunto de teste**: acurácia, matriz de confusão, precisão, recall e F1.
7. **Retreino com todas as 286 instâncias** e gravação em `modelos/modelo.joblib`.

Opções: `--dados`, `--saida`, `--teste 0.2`, `--folds 10`, `--repeticoes 3`.

### Resultados obtidos

| Modelo | Acurácia CV (média ± dp) | Acurácia teste |
|---|---|---|
| **Árvore de Decisão** (gini, max_depth=2) | **0,7723 ± 0,0556** | 0,7069 |
| AdaBoost | 0,7590 ± 0,0565 | 0,7069 |
| KNN | 0,7515 ± 0,0636 | 0,7241 |
| Gradient Boosting | 0,7474 ± 0,0581 | 0,7069 |
| SVM | 0,7458 ± 0,0474 | 0,7069 |
| Random Forest | 0,7443 ± 0,0511 | 0,7241 |
| Rede Neural (MLP) | 0,7352 ± 0,0725 | 0,6724 |
| Regressão Logística | 0,7283 ± 0,0368 | 0,7241 |
| Naive Bayes | 0,7250 ± 0,0796 | 0,6724 |
| Baseline (classe majoritária) | 0,7020 ± 0,0126 | 0,7069 |

O melhor modelo é uma **árvore de decisão rasa (profundidade 2)**, que divide principalmente por
`deg-malig` e `inv-nodes`/`node-caps`, os atributos com maior associação à recorrência na análise exploratória.

**Observações:**
- A base é pequena e ruidosa. Na literatura, a acurácia típica fica entre 70% e 78%, e o baseline já alcança cerca de 70%.
- O teste tem só 58 pacientes, então cada erro vale cerca de 1,7 ponto percentual. Por isso a seleção usa a validação cruzada repetida, que é uma estimativa mais estável.
- O recall da classe `recurrence-events` é baixo, efeito do desbalanceamento quando se otimiza apenas a acurácia.

## 2. Módulo de inferência

Recebe uma nova paciente e devolve o indicativo de diagnóstico (`Class`) e a probabilidade de recorrência.

```bash
# argumentos
python inferencia.py --age 40-49 --menopause premeno --tumor-size 30-34 --inv-nodes 6-8 \
  --node-caps yes --deg-malig 3 --breast left --breast-quad left_low --irradiat yes

# arquivo JSON (uma paciente ou uma lista)
python inferencia.py --json exemplos/paciente.json
python inferencia.py --json exemplos/pacientes.json --saida-json

# modo interativo
python inferencia.py
```

Uso como biblioteca:

```python
from inferencia import Diagnosticador
d = Diagnosticador()
d.prever({"age": "50-59", "menopause": "ge40", "tumor-size": "15-19", "inv-nodes": "0-2",
          "node-caps": "no", "deg-malig": "1", "breast": "right",
          "breast-quad": "central", "irradiat": "no"})
# {'Class': 'no-recurrence-events', 'probabilidade_recorrencia': 0.2378, 'confianca': 0.7622}
```

Os valores são validados contra o domínio de cada atributo. Use `?` para valor desconhecido; ele é
imputado pela moda aprendida no treino.

| Atributo | Valores |
|---|---|
| age | 10-19, 20-29, …, 90-99 |
| menopause | lt40, ge40, premeno |
| tumor-size | 0-4, 5-9, …, 55-59 |
| inv-nodes | 0-2, 3-5, …, 36-39 |
| node-caps | yes, no |
| deg-malig | 1, 2, 3 |
| breast | left, right |
| breast-quad | left_up, left_low, right_up, right_low, central |
| irradiat | yes, no |

> Projeto acadêmico. Não substitui avaliação médica.
