"""Confere se o ambiente reproduz os resultados de referência dos
projetos-teste.

Uso (com o ambiente ativado):
    python verificar_instalacao.py
"""

import os
import sys
import warnings
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from classe_pert_universal import PERTUniversalEngine  # noqa: E402

PASTA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "projetos_teste")

# Média do projeto (dias) com 50.000 simulações e semente 42
REFERENCIA = {
    "projeto1_condicional": {
        "Beta": 38.77,
        "Triangular": 39.23,
        "Lognormal": 39.88,
        "Weibull": 39.58,
        "PERT_Tradicional": 38.33,
    },
    "projeto2_aldo_mattos": {
        "Beta": 28.00,
        "Triangular": 28.00,
        "Lognormal": 29.45,
        "Weibull": 29.00,
        "PERT_Tradicional": 28.00,
    },
}


def main():
    falhas = 0
    for projeto, esperado in REFERENCIA.items():
        base = os.path.join(PASTA, projeto)
        engine = PERTUniversalEngine(nome_projeto=projeto)
        engine.carregar_dados(
            f"{base}_atividades.csv",
            f"{base}_correlacoes.csv",
            f"{base}_condicoes.csv",
        )
        comparacao = engine.comparar_com_pert_tradicional(
            n_simulacoes=50000, seed=42
        )
        df = comparacao["resumo"]
        medias = df.set_index("Métrica / Percentil").loc["Média do Projeto"]
        for dist, valor in esperado.items():
            obtido = float(medias[dist])
            ok = abs(obtido - valor) < 0.011
            falhas += not ok
            marca = "OK  " if ok else "ERRO"
            print(f"[{marca}] {projeto:22s} {dist:17s} "
                  f"esperado {valor:6.2f}  obtido {obtido:6.2f}")

        duracoes = comparacao["duracoes"]
        metricas = [
            "R quantis",
            "KS (D)",
            "Wasserstein-1 (dias)",
            "KL(sim || PERT) (nats)",
            "KL(PERT || sim) (nats)",
            "Jensen-Shannon (nats)",
        ]
        if not all(
            math.isfinite(float(valor))
            for valor in duracoes[metricas].to_numpy().ravel()
        ):
            print(f"[ERRO] {projeto}: métrica de duração não finita.")
            falhas += 1

        soma_criticidades = comparacao["caminhos"].groupby(
            "Distribuição"
        )["Criticidade Monte Carlo (%)"].sum()
        if not all(abs(float(valor) - 100.0) <= 0.05
                   for valor in soma_criticidades):
            print(f"[ERRO] {projeto}: criticidades não somam 100%.")
            falhas += 1
        else:
            print(f"[OK  ] {projeto:22s} métricas e criticidades conferidas")

        for plotar in (
            engine.plotar_grafo,
            engine.plotar_curva_s_comparativa,
        ):
            figura = plotar(mostrar=False)
            plt.close(figura)

        figura = engine.plotar_histograma_comparativo(mostrar=False)
        rotulos = {linha.get_label() for linha in figura.axes[0].get_lines()}
        if "PERT tradicional (Normal)" not in rotulos:
            print(f"[ERRO] {projeto}: curva do PERT ausente no histograma.")
            falhas += 1
        else:
            print(f"[OK  ] {projeto:22s} curva do PERT no histograma")
        plt.close(figura)

    print()
    if falhas:
        print(f"{falhas} resultado(s) diferente(s) da referência.")
        sys.exit(1)
    print("Instalação verificada: todos os resultados conferem.")


if __name__ == "__main__":
    main()
