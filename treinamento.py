"""
Módulo de TREINAMENTO — Breast Cancer (UCI #14)

Etapas:
 1. Carga e análise exploratória (dimensões, ausentes, distribuição da classe)
 2. Divisão estratificada treino/teste (holdout 80/20)
 3. Pipeline de pré-processamento (codificação, imputação, padronização, one-hot)
 4. Comparação de vários classificadores com validação cruzada estratificada
    e busca de hiperparâmetros (GridSearchCV), otimizando ACURÁCIA
 5. Seleção do modelo de melhor acurácia na validação cruzada
 6. Avaliação no conjunto de teste (acurácia, matriz de confusão, relatório)
 7. Retreino do melhor modelo com todos os dados e persistência (joblib)

Uso:
    python treinamento.py
    python treinamento.py --dados dados/breast-cancer.csv --saida modelos/modelo.joblib
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.preprocessamento import (ALVO, ATRIBUTOS, CLASSE_POSITIVA,  # noqa: E402
                                  carregar_dados, codificar,
                                  construir_preprocessador)

warnings.filterwarnings("ignore")
SEMENTE = 42
RAIZ = Path(__file__).resolve().parent


def montar_pipeline(classificador):
    return Pipeline([
        ("codificar", FunctionTransformer(codificar)),
        ("preprocessar", construir_preprocessador()),
        ("modelo", classificador),
    ])


def candidatos():
    """Modelos vistos em sala + grade de hiperparâmetros de cada um."""
    return {
        "Baseline (classe majoritária)": (DummyClassifier(strategy="most_frequent"), {}),
        "Naive Bayes": (GaussianNB(), {"modelo__var_smoothing": [1e-9, 1e-3, 1e-1, 1.0]}),
        "KNN": (KNeighborsClassifier(), {
            "modelo__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
            "modelo__weights": ["uniform", "distance"],
            "modelo__p": [1, 2],
        }),
        "Regressão Logística": (LogisticRegression(max_iter=2000), {
            "modelo__C": [0.01, 0.1, 1, 10],
        }),
        "Árvore de Decisão": (DecisionTreeClassifier(random_state=SEMENTE), {
            "modelo__criterion": ["gini", "entropy"],
            "modelo__max_depth": [2, 3, 4, 5, 8, None],
            "modelo__min_samples_leaf": [1, 5, 10],
        }),
        "SVM": (SVC(probability=True, random_state=SEMENTE), {
            "modelo__kernel": ["linear", "rbf"],
            "modelo__C": [0.1, 1, 10],
            "modelo__gamma": ["scale", 0.01, 0.1],
        }),
        "Random Forest": (RandomForestClassifier(random_state=SEMENTE, n_jobs=-1), {
            "modelo__n_estimators": [100, 300],
            "modelo__max_depth": [3, 5, None],
            "modelo__min_samples_leaf": [1, 5],
        }),
        "AdaBoost": (AdaBoostClassifier(random_state=SEMENTE), {
            "modelo__n_estimators": [50, 100, 200],
            "modelo__learning_rate": [0.05, 0.1, 0.5, 1.0],
        }),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=SEMENTE), {
            "modelo__n_estimators": [50, 100],
            "modelo__learning_rate": [0.05, 0.1],
            "modelo__max_depth": [2, 3],
        }),
        "Rede Neural (MLP)": (MLPClassifier(max_iter=2000, random_state=SEMENTE), {
            "modelo__hidden_layer_sizes": [(8,), (16,), (16, 8)],
            "modelo__alpha": [1e-3, 1e-1, 1.0],
        }),
    }


def analise_exploratoria(X, y):
    print("=" * 70)
    print("1. ANÁLISE EXPLORATÓRIA")
    print("=" * 70)
    print(f"Instâncias: {len(X)} | Atributos: {X.shape[1]}")
    ausentes = X.isna().sum()
    print("\nValores ausentes ('?'):")
    print(ausentes[ausentes > 0].to_string() if ausentes.any() else "  nenhum")
    print("\nDistribuição da classe:")
    for rotulo, n in [("no-recurrence-events", (y == 0).sum()), (CLASSE_POSITIVA, (y == 1).sum())]:
        print(f"  {rotulo:22s} {n:4d} ({n / len(y):.1%})")
    print(f"\nAcurácia de referência (sempre prever a classe majoritária): {max(y.mean(), 1 - y.mean()):.2%}")
    print("\nTaxa de recorrência por atributo (valores mais relevantes):")
    df = X.assign(_y=y)
    for col in ["deg-malig", "inv-nodes", "node-caps", "tumor-size", "irradiat"]:
        taxa = df.groupby(col)["_y"].agg(["mean", "count"]).rename(columns={"mean": "taxa_rec", "count": "n"})
        print(f"\n  {col}:\n" + "\n".join("    " + l for l in taxa.round(2).to_string().splitlines()))


def main():
    ap = argparse.ArgumentParser(description="Treina e seleciona o melhor modelo (acurácia).")
    ap.add_argument("--dados", default=str(RAIZ / "dados" / "breast-cancer.csv"))
    ap.add_argument("--saida", default=str(RAIZ / "modelos" / "modelo.joblib"))
    ap.add_argument("--teste", type=float, default=0.2, help="Fração do holdout de teste")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--repeticoes", type=int, default=3, help="Repetições da validação cruzada")
    args = ap.parse_args()

    X, y = carregar_dados(args.dados)
    analise_exploratoria(X, y)

    # ---- 2. Divisão treino/teste ------------------------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.teste, stratify=y, random_state=SEMENTE)
    print("\n" + "=" * 70)
    print("2. DIVISÃO TREINO / TESTE (estratificada)")
    print("=" * 70)
    print(f"Treino: {len(X_tr)} | Teste: {len(X_te)}")

    # ---- 3/4. Validação cruzada + GridSearch -------------------------------------
    cv = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeticoes, random_state=SEMENTE)
    print("\n" + "=" * 70)
    print(f"3. COMPARAÇÃO DE MODELOS — {args.folds}-fold estratificado x {args.repeticoes} repetições (métrica: acurácia)")
    print("=" * 70)

    resultados = []
    buscas = {}
    for nome, (clf, grade) in candidatos().items():
        t0 = time.time()
        busca = GridSearchCV(montar_pipeline(clf), grade or {"modelo__strategy": ["most_frequent"]},
                             scoring="accuracy", cv=cv, n_jobs=-1, refit=True)
        busca.fit(X_tr, y_tr)
        i = busca.best_index_
        media = busca.cv_results_["mean_test_score"][i]
        dp = busca.cv_results_["std_test_score"][i]
        acc_teste = accuracy_score(y_te, busca.predict(X_te))
        buscas[nome] = busca
        params = {k.replace("modelo__", ""): v for k, v in busca.best_params_.items()}
        resultados.append({"modelo": nome, "acc_cv": media, "dp_cv": dp,
                           "acc_teste": acc_teste, "parametros": params})
        print(f"  {nome:30s} CV = {media:.4f} ± {dp:.4f} | teste = {acc_teste:.4f} | {time.time() - t0:5.1f}s")

    tabela = pd.DataFrame(resultados).sort_values("acc_cv", ascending=False).reset_index(drop=True)

    # ---- 5. Seleção -------------------------------------------------------------
    # Critério: maior acurácia média na validação cruzada (o teste fica reservado
    # para estimar o desempenho sem viés de seleção).
    melhor = tabela.iloc[0]
    nome_melhor = melhor["modelo"]
    busca_melhor = buscas[nome_melhor]
    print("\n" + "=" * 70)
    print("4. RANKING (ordenado pela acurácia na validação cruzada)")
    print("=" * 70)
    print(tabela[["modelo", "acc_cv", "dp_cv", "acc_teste"]].round(4).to_string(index=False))
    print(f"\n>>> MELHOR MODELO: {nome_melhor}")
    print(f"    Hiperparâmetros: {melhor['parametros']}")

    # ---- 6. Avaliação no teste ---------------------------------------------------
    y_pred = busca_melhor.predict(X_te)
    cm = confusion_matrix(y_te, y_pred)
    print("\n" + "=" * 70)
    print("5. AVALIAÇÃO DO MELHOR MODELO NO CONJUNTO DE TESTE")
    print("=" * 70)
    print(f"Acurácia: {accuracy_score(y_te, y_pred):.4f}")
    print("\nMatriz de confusão (linhas = real, colunas = previsto):")
    print(pd.DataFrame(cm, index=["real: no-rec", "real: rec"], columns=["prev: no-rec", "prev: rec"]).to_string())
    print("\n" + classification_report(y_te, y_pred, target_names=["no-recurrence-events", CLASSE_POSITIVA], digits=4))

    # ---- 7. Retreino com todos os dados e persistência -----------------------------
    final = clone(busca_melhor.best_estimator_).fit(X, y)
    Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
    pacote = {
        "pipeline": final,
        "nome_modelo": nome_melhor,
        "hiperparametros": melhor["parametros"],
        "acuracia_cv": float(melhor["acc_cv"]),
        "desvio_cv": float(melhor["dp_cv"]),
        "acuracia_teste": float(accuracy_score(y_te, y_pred)),
        "atributos": ATRIBUTOS,
        "classes": {0: "no-recurrence-events", 1: CLASSE_POSITIVA},
        "versao_sklearn": sklearn.__version__,
        "treinado_em": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    joblib.dump(pacote, args.saida)

    pasta_res = RAIZ / "resultados"
    pasta_res.mkdir(exist_ok=True)
    tabela.assign(parametros=tabela["parametros"].astype(str)).to_csv(pasta_res / "comparacao_modelos.csv", index=False)
    with open(pasta_res / "melhor_modelo.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in pacote.items() if k != "pipeline"}, f, ensure_ascii=False, indent=2, default=str)

    print("=" * 70)
    print(f"Modelo final (retreinado com as {len(X)} instâncias) salvo em: {args.saida}")
    print(f"Resultados salvos em: {pasta_res}/")


if __name__ == "__main__":
    main()
