"""
Módulo de INFERÊNCIA — recebe uma nova paciente e retorna o indicativo de
diagnóstico (atributo Class): no-recurrence-events ou recurrence-events.

Formas de uso:

  1) Argumentos de linha de comando
     python inferencia.py --age 40-49 --menopause premeno --tumor-size 30-34 \
         --inv-nodes 3-5 --node-caps yes --deg-malig 3 --breast left \
         --breast-quad left_low --irradiat yes

  2) Arquivo JSON (uma paciente ou lista de pacientes)
     python inferencia.py --json exemplos/paciente.json

  3) Modo interativo (sem argumentos)
     python inferencia.py

  4) Como biblioteca
     from inferencia import Diagnosticador
     d = Diagnosticador()
     d.prever({"age": "50-59", "menopause": "ge40", ...})

Valores ausentes podem ser informados como "?" (serão imputados pela moda do treino).
"""
import argparse
import json
import sys
from pathlib import Path

import joblib

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
from src.preprocessamento import ATRIBUTOS, DOMINIOS, validar_paciente  # noqa: E402

MODELO_PADRAO = RAIZ / "modelos" / "modelo.joblib"


class Diagnosticador:
    def __init__(self, caminho_modelo=MODELO_PADRAO):
        caminho_modelo = Path(caminho_modelo)
        if not caminho_modelo.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em {caminho_modelo}. Execute antes: python treinamento.py")
        self.pacote = joblib.load(caminho_modelo)
        self.pipeline = self.pacote["pipeline"]
        self.classes = self.pacote["classes"]

    def prever(self, paciente):
        X = validar_paciente(paciente)
        pred = int(self.pipeline.predict(X)[0])
        resultado = {"Class": self.classes[pred]}
        if hasattr(self.pipeline, "predict_proba"):
            proba = self.pipeline.predict_proba(X)[0]
            resultado["probabilidade_recorrencia"] = round(float(proba[1]), 4)
            resultado["confianca"] = round(float(proba[pred]), 4)
        return resultado


def ler_interativo():
    print("Informe os dados da paciente ('?' para valor desconhecido):\n")
    paciente = {}
    for atr in ATRIBUTOS:
        while True:
            v = input(f"  {atr} {DOMINIOS[atr]}: ").strip()
            if v == "?" or v in DOMINIOS[atr]:
                paciente[atr] = v
                break
            print("    valor inválido, tente novamente.")
    return paciente


def imprimir(paciente, res, info):
    print("-" * 60)
    for atr in ATRIBUTOS:
        print(f"  {atr:12s}: {paciente.get(atr, '?')}")
    print("-" * 60)
    print(f"  Indicativo (Class): {res['Class'].upper()}")
    if "probabilidade_recorrencia" in res:
        print(f"  Probabilidade de recorrência: {res['probabilidade_recorrencia']:.1%}")
        print(f"  Confiança na previsão:        {res['confianca']:.1%}")
    print(f"  Modelo: {info['nome_modelo']} (acurácia CV {info['acuracia_cv']:.2%}, "
          f"teste {info['acuracia_teste']:.2%})")
    print("  Aviso: ferramenta acadêmica; não substitui avaliação médica.")
    print("-" * 60)


def main():
    ap = argparse.ArgumentParser(description="Indicativo de diagnóstico (Class) para nova paciente.")
    ap.add_argument("--modelo", default=str(MODELO_PADRAO))
    ap.add_argument("--json", help="Arquivo JSON com uma paciente (objeto) ou várias (lista)")
    ap.add_argument("--saida-json", action="store_true", help="Imprime apenas o resultado em JSON")
    for atr in ATRIBUTOS:
        ap.add_argument(f"--{atr}", dest=atr.replace("-", "_"), choices=DOMINIOS[atr] + ["?"])
    args = ap.parse_args()

    d = Diagnosticador(args.modelo)

    if args.json:
        with open(args.json, encoding="utf-8") as f:
            dados = json.load(f)
        pacientes = dados if isinstance(dados, list) else [dados]
    else:
        informados = {a: getattr(args, a.replace("-", "_")) for a in ATRIBUTOS}
        if all(v is None for v in informados.values()):
            pacientes = [ler_interativo()]
        else:
            faltando = [a for a, v in informados.items() if v is None]
            if faltando:
                ap.error(f"atributos não informados: {faltando} (use '?' para desconhecido)")
            pacientes = [informados]

    saidas = []
    for p in pacientes:
        try:
            res = d.prever(p)
        except ValueError as e:
            print(f"Erro nos dados da paciente:\n{e}", file=sys.stderr)
            sys.exit(1)
        saidas.append({**p, **res})
        if not args.saida_json:
            imprimir(p, res, d.pacote)
    if args.saida_json:
        print(json.dumps(saidas if len(saidas) > 1 else saidas[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
