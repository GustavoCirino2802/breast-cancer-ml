"""
Pré-processamento compartilhado entre treinamento e inferência.

Base: UCI Breast Cancer (https://archive.ics.uci.edu/dataset/14/breast+cancer)
286 pacientes, 9 atributos categóricos, alvo binário `Class`.

Estratégia:
- Atributos em faixas (age, tumor-size, inv-nodes) e deg-malig são ORDINAIS:
  convertidos para o ponto médio da faixa (preserva a ordem e a escala).
- node-caps, breast e irradiat são BINÁRIOS: 0/1.
- menopause e breast-quad são NOMINAIS: one-hot encoding.
- Valores ausentes ("?" -> NaN): imputação pela moda (aprendida só no treino).
- Atributos numéricos padronizados (StandardScaler), essencial para KNN/SVM/LR/MLP.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ALVO = "Class"
CLASSE_POSITIVA = "recurrence-events"

ATRIBUTOS = [
    "age", "menopause", "tumor-size", "inv-nodes", "node-caps",
    "deg-malig", "breast", "breast-quad", "irradiat",
]

# Domínios válidos de cada atributo (conforme breast-cancer.names)
DOMINIOS = {
    "age": ["10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99"],
    "menopause": ["lt40", "ge40", "premeno"],
    "tumor-size": ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34",
                   "35-39", "40-44", "45-49", "50-54", "55-59"],
    "inv-nodes": ["0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20",
                  "21-23", "24-26", "27-29", "30-32", "33-35", "36-39"],
    "node-caps": ["yes", "no"],
    "deg-malig": ["1", "2", "3"],
    "breast": ["left", "right"],
    "breast-quad": ["left_up", "left_low", "right_up", "right_low", "central"],
    "irradiat": ["yes", "no"],
}

ORDINAIS_FAIXA = ["age", "tumor-size", "inv-nodes"]
BINARIOS = {"node-caps": "yes", "breast": "right", "irradiat": "yes"}
NOMINAIS = ["menopause", "breast-quad"]
NUMERICOS = ORDINAIS_FAIXA + ["deg-malig"] + list(BINARIOS)


def carregar_dados(caminho):
    """Lê o CSV, troca '?' por NaN e separa X / y (1 = recorrência)."""
    df = pd.read_csv(caminho, na_values="?", dtype=str, skipinitialspace=True)
    df = df.dropna(how="all")
    df.columns = [c.strip() for c in df.columns]
    X = df[ATRIBUTOS].copy()
    y = (df[ALVO].str.strip() == CLASSE_POSITIVA).astype(int)
    return X, y


def _ponto_medio(faixa):
    if pd.isna(faixa):
        return np.nan
    a, b = str(faixa).split("-")
    return (float(a) + float(b)) / 2


def codificar(X):
    """Converte strings categóricas em colunas numéricas/categóricas limpas.

    Função de módulo (e não lambda) para que o pipeline possa ser serializado.
    """
    X = pd.DataFrame(X, columns=ATRIBUTOS).copy()
    out = pd.DataFrame(index=X.index)
    for col in ORDINAIS_FAIXA:
        out[col] = X[col].map(_ponto_medio)
    out["deg-malig"] = pd.to_numeric(X["deg-malig"], errors="coerce")
    for col, positivo in BINARIOS.items():
        out[col] = X[col].map(lambda v, p=positivo: np.nan if pd.isna(v) else float(v == p))
    for col in NOMINAIS:
        out[col] = X[col].astype(object)
    return out


def construir_preprocessador():
    """ColumnTransformer: imputação + padronização (numéricos) e imputação + one-hot (nominais)."""
    numerico = Pipeline([
        ("imputador", SimpleImputer(strategy="most_frequent")),
        ("escala", StandardScaler()),
    ])
    nominal = Pipeline([
        ("imputador", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(categories=[DOMINIOS[c] for c in NOMINAIS],
                                 handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numerico, NUMERICOS),
        ("nom", nominal, NOMINAIS),
    ])


def validar_paciente(paciente):
    """Confere se os valores de uma nova paciente pertencem ao domínio. '?' / None = ausente."""
    erros = []
    limpo = {}
    for atr in ATRIBUTOS:
        valor = paciente.get(atr)
        if valor is None or str(valor).strip() in ("", "?"):
            limpo[atr] = np.nan
            continue
        valor = str(valor).strip()
        if valor not in DOMINIOS[atr]:
            erros.append(f"'{atr}' = '{valor}' inválido. Valores aceitos: {DOMINIOS[atr]}")
        limpo[atr] = valor
    if erros:
        raise ValueError("\n".join(erros))
    return pd.DataFrame([limpo], columns=ATRIBUTOS)
