import re
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import brentq, fsolve
from scipy.special import gamma as gamma_func
from scipy.spatial.distance import jensenshannon
from scipy.stats import beta, gaussian_kde, ks_2samp, kstest, lognorm, norm, triang


class PERTUniversalEngine:
    """Engine unificada de Simulação Monte Carlo PERT/CPM.

    Suporta Cópula Gaussiana (Correlação), Regras Condicionais e Múltiplas
    Distribuições: Beta (moda M, média ponderada TE), Triangular, Lognormal
    deslocada, Weibull de 3 parâmetros e Normal ancorada na moda (média = M,
    σ = (P - O) / 6).

    A Normal é um baseline ingênuo (contraponto didático) no comparativo:
    sendo simétrica, ela ignora a assimetria típica de durações de projeto
    e não é recomendada para modelar atividades reais; serve para mostrar
    o quanto essa simplificação distorce percentis altos (P90/P95) frente
    às distribuições assimétricas (Beta, Triangular, Lognormal, Weibull).
    """

    DISTRIBUICOES = ["Beta", "Triangular", "Lognormal", "Weibull", "Normal"]
    # Nomes aceitos (minúsculos) -> nome usado nas tabelas e gráficos
    ALIASES = {
        "beta": "Beta", "beta_pert": "Beta", "pert": "Beta",
        "triangular": "Triangular", "triang": "Triangular",
        "lognormal": "Lognormal", "lognorm": "Lognormal",
        "weibull": "Weibull", "weibull_min": "Weibull",
        "normal": "Normal", "gaussiana": "Normal", "gauss": "Normal",
    }
    PERCENTIS = [10, 25, 50, 75, 80, 85, 90, 95]
    ESTILOS = {
        "Beta": ("#1f77b4", "-"),
        "Triangular": ("#ff7f0e", "--"),
        "Lognormal": ("#2ca02c", "-."),
        "Weibull": ("#d62728", ":"),
        "Normal": ("#9467bd", (0, (5, 1))),
        "PERT_Tradicional": ("#000000", (0, (1, 1))),
    }

    def __init__(
        self,
        nome_projeto="Projeto PERT/CPM",
        p_pessimista_weibull=0.99,
    ):
        self.nome_projeto = nome_projeto
        self.p_pessimista_weibull = p_pessimista_weibull
        self._resetar_rede()

    @classmethod
    def _nome_distribuicao(cls, nome):
        """Converte apelidos ("Beta_PERT", "pert", "triang"...) no nome
        padrão ("Beta", "Triangular", "Lognormal", "Weibull")."""
        padrao = cls.ALIASES.get(str(nome).strip().lower())
        if padrao is None:
            raise ValueError(f"Distribuição '{nome}' não suportada.")
        return padrao

    def _resetar_rede(self):
        self.grafo = nx.DiGraph()
        self.atividades = {}
        self.regras_condicionais = {}
        # Condições definidas por dados (ver adicionar_condicao)
        self.condicoes = {}
        self.matriz_correlacao = None
        self._resultados_cache = {}
        # Durações do projeto por distribuição: {dist: array de durações}
        self.resultados = {}
        self.resultado_pert_tradicional = None

    def _limpar_resultados(self):
        """Descarta resultados de simulações feitas com a rede anterior."""
        self._resultados_cache = {}
        self.resultados = {}
        self.resultado_pert_tradicional = None

    # =========================================================================
    # CÁLCULOS E AMOSTRAGEM ESTATÍSTICA (INTEGRADOS)
    # =========================================================================
    def _calcular_parametros(self, O, M, P):
        if not (O <= M <= P):
            raise ValueError(
                f"Estimativas inválidas: O ({O}) <= M ({M}) <= P ({P}) não respeitado."
            )

        O, M, P = float(O), float(M), float(P)
        params = {}

        if O == M == P:
            return {
                "constante": True,
                "valor": O,
                "Beta": {"TE": O, "Var": 0.0},
                "Triangular": {"TE": O, "Var": 0.0},
                "Lognormal": {"TE": O, "Var": 0.0},
                "Weibull": {"TE": O, "Var": 0.0},
                "Normal": {"TE": O, "Var": 0.0},
            }

        scale_range = P - O

        # 1. Beta analítica parametrizada por O, M e P, com moda em M
        te_beta = (O + 4.0 * M + P) / 6.0
        var_beta = (te_beta - O) * (P - te_beta) / 7.0
        params["Beta"] = {
            "TE": te_beta,
            "Var": var_beta,
            "alpha": 1.0 + 4.0 * (M - O) / scale_range,
            "beta": 1.0 + 4.0 * (P - M) / scale_range,
            "loc": O,
            "scale": scale_range,
        }

        # 2. Triangular
        te_tri = (O + M + P) / 3.0
        var_tri = (
            (O - M) ** 2 + (O - P) ** 2 + (M - P) ** 2
        ) / 18.0
        params["Triangular"] = {
            "TE": te_tri,
            "Var": var_tri,
            "c": (M - O) / scale_range,
            "loc": O,
            "scale": scale_range,
        }

        # 3. Lognormal Deslocada
        sigma_log = (np.log(P - O) - np.log(M - O)) / 3.0
        sigma_log = max(0.01, min(sigma_log, 1.2))
        lambda_param = np.log(M - O) + sigma_log**2
        scale_log = np.exp(lambda_param)
        te_log = O + (M - O) * np.exp(1.5 * sigma_log**2)
        var_log = (
            ((M - O) ** 2)
            * np.exp(3.0 * sigma_log**2)
            * (np.exp(sigma_log**2) - 1.0)
        )
        params["Lognormal"] = {
            "TE": te_log,
            "Var": var_log,
            "sigma": sigma_log,
            "scale": scale_log,
            "loc": O,
        }

        # 4. Weibull de 3 Parâmetros
        epsilon = O
        m_rel = (M - epsilon) / (P - epsilon)

        def equacao_beta(b_val):
            b = b_val[0] if isinstance(b_val, np.ndarray) else b_val
            if b <= 1.0:
                return 1e6
            pico_rel = (1.0 - 1.0 / b) ** (1.0 / b)
            cauda_rel = (
                -np.log(1.0 - self.p_pessimista_weibull)
            ) ** (1.0 / b)
            return (pico_rel / cauda_rel) - m_rel

        # brentq garante convergência quando há raiz em (1, 200); o fsolve
        # partindo de x0=3 travava para M próximo de O (ex.: O=3, M=4, P=9).
        try:
            beta_opt = float(brentq(equacao_beta, 1.0 + 1e-6, 200.0))
        except ValueError:
            try:
                beta_opt = float(fsolve(equacao_beta, x0=3.0)[0])
            except Exception:
                beta_opt = 2.0

        pico_fator = (1.0 - 1.0 / beta_opt) ** (1.0 / beta_opt)
        u_1_opt = epsilon + (M - epsilon) / pico_fator
        f_g1 = gamma_func(1.0 + 1.0 / beta_opt)
        f_g2 = gamma_func(1.0 + 2.0 / beta_opt)
        delta_u = u_1_opt - epsilon

        params["Weibull"] = {
            "TE": epsilon + delta_u * f_g1,
            "Var": (delta_u**2) * (f_g2 - f_g1**2),
            "epsilon": epsilon,
            "u_1": u_1_opt,
            "beta": beta_opt,
            "delta_u": delta_u,
        }

        # 5. Normal ancorada na moda: média = M (e não a TE ponderada),
        # desvio padrão pela regra clássica do PERT, σ = (P - O) / 6.
        sigma_normal = scale_range / 6.0
        params["Normal"] = {
            "TE": M,
            "Var": sigma_normal**2,
            "mu": M,
            "sigma": sigma_normal,
        }

        return {"constante": False, **params}

    def gerar_amostras(
        self,
        distrib_name,
        O=None,
        M=None,
        P=None,
        n_sim=50000,
        u_quantil=None,
        calc_params=None,
    ):
        """Gera amostras para uma distribuição (com ou sem quantis de
        correlação)."""
        if calc_params is None:
            calc_params = self._calcular_parametros(O, M, P)

        if calc_params.get("constante", False):
            return np.full(n_sim, float(calc_params["valor"]))

        distrib = self._nome_distribuicao(distrib_name)

        if distrib == "Beta":
            p = calc_params["Beta"]
            if u_quantil is not None:
                return beta.ppf(
                    u_quantil,
                    p["alpha"],
                    p["beta"],
                    loc=p["loc"],
                    scale=p["scale"],
                )
            return p["loc"] + p["scale"] * np.random.beta(
                p["alpha"], p["beta"], size=n_sim
            )

        elif distrib == "Triangular":
            p = calc_params["Triangular"]
            if u_quantil is not None:
                return triang.ppf(
                    u_quantil, p["c"], loc=p["loc"], scale=p["scale"]
                )
            return np.random.triangular(
                left=p["loc"],
                mode=p["loc"] + p["c"] * p["scale"],
                right=p["loc"] + p["scale"],
                size=n_sim,
            )

        elif distrib == "Lognormal":
            p = calc_params["Lognormal"]
            if u_quantil is not None:
                return p["loc"] + lognorm.ppf(
                    u_quantil, s=p["sigma"], scale=p["scale"]
                )
            return p["loc"] + lognorm.rvs(
                s=p["sigma"], scale=p["scale"], size=n_sim
            )

        elif distrib == "Weibull":
            p = calc_params["Weibull"]
            if p["delta_u"] <= 0:
                return np.full(n_sim, float(p["epsilon"]))

            if u_quantil is None:
                u_quantil = np.random.uniform(
                    1e-10, 1.0 - 1e-10, size=n_sim
                )
            else:
                u_quantil = np.clip(u_quantil, 1e-10, 1.0 - 1e-10)

            return p["epsilon"] + p["delta_u"] * (
                -np.log(1.0 - u_quantil)
            ) ** (1.0 / p["beta"])

        elif distrib == "Normal":
            p = calc_params["Normal"]
            if u_quantil is not None:
                return norm.ppf(u_quantil, loc=p["mu"], scale=p["sigma"])
            return np.random.normal(loc=p["mu"], scale=p["sigma"], size=n_sim)

        raise ValueError(
            f"Distribuição '{distrib_name}' não suportada."
        )

    # =========================================================================
    # CONFIGURAÇÃO DA REDE E SIMULAÇÃO
    # =========================================================================
    def adicionar_atividade(
        self,
        nome,
        O=None,
        M=None,
        P=None,
        distrib="Beta",
        predecessoras=None,
        descricao="",
    ):
        self.grafo.add_node(nome, descricao=descricao)

        calc_params = {}
        if None not in (O, M, P):
            calc_params = self._calcular_parametros(O, M, P)

        self.atividades[nome] = {
            "O": O,
            "M": M,
            "P": P,
            "distrib": distrib,
            "calc_params": calc_params,
            "descricao": descricao,
            "predecessoras": list(predecessoras or []),
        }

        if predecessoras:
            for pred in predecessoras:
                self.grafo.add_edge(pred, nome)

        self._limpar_resultados()

    def definir_matriz_correlacao(self, df_corr):
        self.matriz_correlacao = df_corr.copy()
        self._limpar_resultados()

    def adicionar_regra_condicional(self, atividade, funcao_regra):
        self.regras_condicionais[atividade] = funcao_regra
        self.condicoes.pop(atividade, None)
        self._limpar_resultados()

    def adicionar_condicao(
        self, atividade, limite_es, cenario_ate, cenario_acima
    ):
        """Regra condicional definida por dados.

        Se o início mais cedo (ES) da atividade for <= limite_es, a duração
        segue cenario_ate (O, M, P); caso contrário, cenario_acima (O, M, P).
        """
        params_ate = self._calcular_parametros(*cenario_ate)
        params_acima = self._calcular_parametros(*cenario_acima)

        def regra(es, n_sim, engine, distrib=None):
            dist = distrib or engine.atividades[atividade]["distrib"]
            amostras_ate = engine.gerar_amostras(
                dist, n_sim=n_sim, calc_params=params_ate
            )
            amostras_acima = engine.gerar_amostras(
                dist, n_sim=n_sim, calc_params=params_acima
            )
            return np.where(es <= limite_es, amostras_ate, amostras_acima)

        self.adicionar_regra_condicional(atividade, regra)
        self.condicoes[atividade] = {
            "limite_es": float(limite_es),
            "cenario_ate": tuple(cenario_ate),
            "cenario_acima": tuple(cenario_acima),
        }

    # =========================================================================
    # ENTRADA DE DADOS POR TABELA
    # =========================================================================
    @staticmethod
    def _numero(valor):
        if valor is None or (isinstance(valor, str) and not valor.strip()):
            return None
        if pd.isna(valor):
            return None
        return float(valor)

    @staticmethod
    def _lista_predecessoras(valor):
        if valor is None or pd.isna(valor):
            return []
        partes = str(valor).replace(",", ";").split(";")
        return [p.strip() for p in partes if p.strip() not in ("", "-")]

    def carregar_dados(
        self, df_atividades, df_correlacao=None, df_condicoes=None
    ):
        """Monta o projeto a partir de tabelas (DataFrame ou caminho CSV).

        Substitui qualquer rede cadastrada antes.

        df_atividades: Atividade, O, M, P, Predecessoras ("B;D")
                       e opcionalmente Descricao, Distribuicao
        df_correlacao: Atividade_1, Atividade_2, Correlacao
        df_condicoes:  Atividade, Limite_ES, O_ate, M_ate, P_ate,
                       O_acima, M_acima, P_acima
        """

        def ler(tabela):
            if tabela is None:
                return None
            if isinstance(tabela, str):
                return pd.read_csv(tabela)
            return tabela

        df_atividades = ler(df_atividades)
        df_correlacao = ler(df_correlacao)
        df_condicoes = ler(df_condicoes)

        self._resetar_rede()

        for _, linha in df_atividades.iterrows():
            nome = str(linha["Atividade"]).strip()
            if not nome or nome == "nan":
                continue
            distrib = linha.get("Distribuicao", "Beta")
            if distrib is None or pd.isna(distrib) or not str(distrib).strip():
                distrib = "Beta"
            descricao = linha.get("Descricao", "")
            self.adicionar_atividade(
                nome,
                O=self._numero(linha.get("O")),
                M=self._numero(linha.get("M")),
                P=self._numero(linha.get("P")),
                distrib=str(distrib).strip(),
                predecessoras=self._lista_predecessoras(
                    linha.get("Predecessoras")
                ),
                descricao="" if pd.isna(descricao) else str(descricao),
            )

        if df_correlacao is not None and len(df_correlacao):
            nomes = list(self.atividades)
            df_corr = pd.DataFrame(
                np.eye(len(nomes)), index=nomes, columns=nomes
            )
            for _, linha in df_correlacao.iterrows():
                a1 = str(linha["Atividade_1"]).strip()
                a2 = str(linha["Atividade_2"]).strip()
                for a in (a1, a2):
                    if a not in self.atividades:
                        raise ValueError(
                            f"Correlação com atividade inexistente: '{a}'"
                        )
                rho = float(linha["Correlacao"])
                if not -1.0 <= rho <= 1.0:
                    raise ValueError(
                        f"Correlação {a1}-{a2} fora de [-1, 1]: {rho}"
                    )
                df_corr.loc[a1, a2] = rho
                df_corr.loc[a2, a1] = rho
            self.definir_matriz_correlacao(df_corr)

        if df_condicoes is not None:
            for _, linha in df_condicoes.iterrows():
                nome = str(linha["Atividade"]).strip()
                if nome not in self.atividades:
                    raise ValueError(
                        f"Condição para atividade inexistente: '{nome}'"
                    )
                self.adicionar_condicao(
                    nome,
                    limite_es=float(linha["Limite_ES"]),
                    cenario_ate=tuple(
                        float(linha[c]) for c in ("O_ate", "M_ate", "P_ate")
                    ),
                    cenario_acima=tuple(
                        float(linha[c])
                        for c in ("O_acima", "M_acima", "P_acima")
                    ),
                )

    def gerar_tabela_premissas(self):
        """Tabela PERT (TE e σ da Beta) das atividades e das condições."""

        def linha(nome, preds, O, M, P):
            te = (O + 4.0 * M + P) / 6.0
            return {
                "Atividade": nome,
                "Predecessoras": ", ".join(preds) or "-",
                "Otimista (O)": O,
                "Mais Provável (M)": M,
                "Pessimista (P)": P,
                "Média (TE)": round(te, 2),
                "Variância (σ²)": round(((P - O) / 6.0) ** 2, 2),
                "Desvio Padrão (σ)": round((P - O) / 6.0, 2),
                # Desvio da Beta analítica parametrizada pela moda M
                "Desvio Beta analítica (moda)": round(
                    float(np.sqrt((te - O) * (P - te) / 7.0)), 2
                ),
                # Normal ancorada na moda: média = M; σ reaproveita a
                # regra clássica (P - O) / 6, acima.
                "Média Normal (moda)": M,
            }

        linhas = []
        for nome, atv in self.atividades.items():
            preds = atv["predecessoras"]
            if nome in self.condicoes:
                c = self.condicoes[nome]
                lim = f"{c['limite_es']:g}"
                linhas.append(
                    linha(f"{nome} (ES <= {lim})", preds, *c["cenario_ate"])
                )
                linhas.append(
                    linha(f"{nome} (ES > {lim})", preds, *c["cenario_acima"])
                )
            elif None not in (atv["O"], atv["M"], atv["P"]):
                linhas.append(
                    linha(nome, preds, atv["O"], atv["M"], atv["P"])
                )
            else:
                linhas.append(
                    {
                        "Atividade": f"{nome} (regra própria)",
                        "Predecessoras": ", ".join(preds) or "-",
                    }
                )
        return pd.DataFrame(linhas)

    def _validar_rede(self):
        nos_reais = [
            n for n in self.grafo.nodes if n not in ("Start", "End")
        ]
        faltando = [n for n in nos_reais if n not in self.atividades]
        if faltando:
            raise ValueError(
                f"Predecessoras não cadastradas como atividade: {faltando}"
            )
        sem_dados = [
            n
            for n in nos_reais
            if not self.atividades[n]["calc_params"]
            and n not in self.regras_condicionais
        ]
        if sem_dados:
            raise ValueError(
                f"Atividades sem O, M, P e sem regra condicional: {sem_dados}"
            )

    def _preparar_nos_virtuais(self):
        # Refaz as ligações com Start/End para refletir a rede atual
        self.grafo.remove_nodes_from(["Start", "End"])
        nos = list(self.grafo.nodes)
        for n in nos:
            if self.grafo.in_degree(n) == 0:
                self.grafo.add_edge("Start", n)
            if self.grafo.out_degree(n) == 0:
                self.grafo.add_edge(n, "End")

    def _gerar_amostras_base(self, n_sim, distrib_override=None):
        # Atividades definidas apenas por regra condicional não têm O, M, P;
        # a duração delas é calculada na regra durante o forward pass.
        nos_reais = [
            n
            for n in self.grafo.nodes
            if n not in ("Start", "End") and self.atividades[n]["calc_params"]
        ]
        amostras = {
            n: np.zeros(n_sim)
            for n in self.grafo.nodes
            if n not in ("Start", "End") and n not in nos_reais
        }

        # Aplicação da Cópula Gaussiana se houver Matriz de Correlação
        if self.matriz_correlacao is not None:
            cols = [
                n for n in nos_reais if n in self.matriz_correlacao.columns
            ]
            if len(cols) > 1:
                corr = self.matriz_correlacao.loc[
                    cols, cols
                ].values.copy()
                np.fill_diagonal(corr, 1.0)

                eigvals, eigvecs = np.linalg.eigh(corr)
                eigvals = np.maximum(eigvals, 1e-8)
                corr = eigvecs @ np.diag(eigvals) @ eigvecs.T

                L = np.linalg.cholesky(corr)
                Z = np.random.normal(size=(n_sim, len(cols)))
                Z_corr = Z @ L.T
                U = norm.cdf(Z_corr)

                for i, no in enumerate(cols):
                    cfg = self.atividades[no]
                    dist = distrib_override or cfg["distrib"]
                    amostras[no] = self.gerar_amostras(
                        distrib_name=dist,
                        n_sim=n_sim,
                        u_quantil=U[:, i],
                        calc_params=cfg["calc_params"],
                    )

        # Preenchimento de nós sem correlação explícita
        for no in nos_reais:
            if no not in amostras:
                cfg = self.atividades[no]
                dist = distrib_override or cfg["distrib"]
                amostras[no] = self.gerar_amostras(
                    distrib_name=dist,
                    n_sim=n_sim,
                    calc_params=cfg["calc_params"],
                )

        amostras["Start"] = np.zeros(n_sim)
        amostras["End"] = np.zeros(n_sim)
        return amostras

    def simular(
        self, n_simulacoes=50000, seed=42, distrib_override=None
    ):
        np.random.seed(seed)
        self._validar_rede()
        self._preparar_nos_virtuais()
        duracoes = self._gerar_amostras_base(
            n_simulacoes, distrib_override=distrib_override
        )

        ordem_topologica = list(nx.topological_sort(self.grafo))
        ES, EF = {}, {}

        # CPM Forward Pass com suporte a Condicionais
        for no in ordem_topologica:
            preds = list(self.grafo.predecessors(no))
            if not preds:
                ES[no] = np.zeros(n_simulacoes)
            else:
                ef_preds = [EF[p] for p in preds]
                ES[no] = (
                    np.maximum.reduce(ef_preds)
                    if len(ef_preds) > 1
                    else ef_preds[0]
                )

            # Execução de Regras Condicionais
            if no in self.regras_condicionais:
                func_regra = self.regras_condicionais[no]
                try:
                    duracoes[no] = func_regra(
                        ES[no],
                        n_simulacoes,
                        engine=self,
                        distrib=distrib_override,
                    )
                except TypeError:
                    duracoes[no] = func_regra(
                        ES[no], n_simulacoes, engine=self
                    )

            EF[no] = ES[no] + duracoes[no]

        caminhos = list(
            nx.all_simple_paths(
                self.grafo, source="Start", target="End"
            )
        )
        matrix_caminhos = np.column_stack(
            [
                sum(duracoes[no] for no in caminho)
                for caminho in caminhos
            ]
        )

        duracao_projeto = np.max(matrix_caminhos, axis=1)
        idx_criticos = np.argmax(matrix_caminhos, axis=1)

        caminhos_nomes = [
            f"Caminho {i+1} ({' -> '.join([n for n in c if n not in ('Start', 'End')])})"
            for i, c in enumerate(caminhos)
        ]

        df_criticidade_caminhos = pd.DataFrame(
            {
                "Caminho": caminhos_nomes,
                "Média (Dias)": np.round(
                    np.mean(matrix_caminhos, axis=0), 2
                ),
                "P90 (Dias)": np.round(
                    np.percentile(matrix_caminhos, 90, axis=0), 2
                ),
                "Criticidade (%)": np.round(
                    [
                        np.sum(idx_criticos == i) / n_simulacoes * 100
                        for i in range(len(caminhos))
                    ],
                    2,
                ),
            }
        ).sort_values(by="Criticidade (%)", ascending=False)

        nos_reais = [
            n for n in self.grafo.nodes if n not in ("Start", "End")
        ]
        df_atividades = pd.DataFrame(
            {
                "Atividade": nos_reais,
                "Duração Média": [
                    round(float(np.mean(duracoes[n])), 2)
                    for n in nos_reais
                ],
                "Desvio Padrão": [
                    round(float(np.std(duracoes[n])), 2)
                    for n in nos_reais
                ],
                "Criticidade Atividade (%)": [
                    round(
                        float(
                            np.mean(
                                [
                                    n in caminhos[idx]
                                    for idx in idx_criticos
                                ]
                            )
                        )
                        * 100,
                        2,
                    )
                    for n in nos_reais
                ],
                # Atividade constante (std = 0) não tem correlação: usa 0
                "Sensibilidade (Corr)": [
                    round(
                        float(
                            np.corrcoef(duracoes[n], duracao_projeto)[
                                0, 1
                            ]
                        ),
                        2,
                    )
                    if np.std(duracoes[n]) > 0
                    else 0.0
                    for n in nos_reais
                ],
            }
        ).sort_values(
            by="Criticidade Atividade (%)", ascending=False
        )

        resumo_lista = [
            {
                "Métrica / Percentil": "Média do Projeto",
                "Valor (Dias)": round(
                    float(np.mean(duracao_projeto)), 2
                ),
            },
            {
                "Métrica / Percentil": "Desvio Padrão",
                "Valor (Dias)": round(
                    float(np.std(duracao_projeto)), 2
                ),
            },
        ]

        for p in self.PERCENTIS:
            resumo_lista.append(
                {
                    "Métrica / Percentil": f"Percentil {p} (P{p})",
                    "Valor (Dias)": round(
                        float(np.percentile(duracao_projeto, p)), 2
                    ),
                }
            )

        df_resumo = pd.DataFrame(resumo_lista)

        return (
            duracao_projeto,
            df_atividades,
            df_criticidade_caminhos,
            df_resumo,
        )

    def simular_todas(
        self,
        n_simulacoes=50000,
        seed=42,
        distribuicoes_teste=None,
        incluir_pert_tradicional=False,
    ):
        """Simula cada distribuição e devolve o resumo lado a lado.

        Com incluir_pert_tradicional=True, acrescenta a coluna
        "PERT_Tradicional" (método analítico, ver pert_tradicional).
        """
        if distribuicoes_teste is None:
            distribuicoes_teste = self.DISTRIBUICOES

        resumo_comparativo = {}

        for dist in distribuicoes_teste:
            dist = self._nome_distribuicao(dist)
            dur_proj, df_ativ, df_cam, df_res = self.simular(
                n_simulacoes=n_simulacoes,
                seed=seed,
                distrib_override=dist,
            )
            self._resultados_cache[dist] = {
                "duracao_projeto": dur_proj,
                "df_atividades": df_ativ,
                "df_caminhos": df_cam,
                "df_resumo": df_res,
            }
            self.resultados[dist] = dur_proj

            dict_res = dict(
                zip(
                    df_res["Métrica / Percentil"],
                    df_res["Valor (Dias)"],
                )
            )
            resumo_comparativo[dist] = dict_res

        if incluir_pert_tradicional:
            df_trad = self.pert_tradicional()
            resumo_comparativo["PERT_Tradicional"] = dict(
                zip(df_trad["Métrica / Percentil"], df_trad["Valor (Dias)"])
            )

        df_comparativo = pd.DataFrame(resumo_comparativo).reset_index()
        df_comparativo.rename(
            columns={"index": "Métrica / Percentil"}, inplace=True
        )
        return df_comparativo

    def analisar_criticidade_caminhos(
        self, dist_escolhida="Beta", n_simulacoes=50000, seed=42
    ):
        dist_escolhida = self._nome_distribuicao(dist_escolhida)
        if dist_escolhida in self._resultados_cache:
            return self._resultados_cache[dist_escolhida]["df_caminhos"]

        _, _, df_caminhos, _ = self.simular(
            n_simulacoes=n_simulacoes,
            seed=seed,
            distrib_override=dist_escolhida,
        )
        return df_caminhos

    def comparar_com_pert_tradicional(
        self,
        n_simulacoes=50000,
        seed=42,
        distribuicoes_teste=None,
        n_bins=200,
        n_quantis=1000,
    ):
        """Compara as durações simuladas e criticidades com o PERT normal.

        O PERT tradicional fornece uma aproximação Normal da duração total e
        um único caminho crítico determinístico. As métricas de distribuição
        comparam cada amostra Monte Carlo com essa Normal; as métricas de
        caminhos comparam a frequência simulada com a escolha determinística.

        Retorna ``{"duracoes": DataFrame, "caminhos": DataFrame}``. R é a
        correlação de Pearson entre funções quantis, não entre observações
        aleatórias pareadas. KL e Jensen-Shannon são aproximadas por histogramas
        comuns com suavização de Jeffreys (0,5 por classe).
        """
        if n_bins < 2:
            raise ValueError("n_bins deve ser pelo menos 2.")
        if n_quantis < 2:
            raise ValueError("n_quantis deve ser pelo menos 2.")
        if distribuicoes_teste is None:
            distribuicoes_teste = self.DISTRIBUICOES

        distribuicoes = [
            self._nome_distribuicao(dist) for dist in distribuicoes_teste
        ]
        df_resumo = self.simular_todas(
            n_simulacoes=n_simulacoes,
            seed=seed,
            distribuicoes_teste=distribuicoes,
            incluir_pert_tradicional=True,
        )
        parametros = self.resultado_pert_tradicional
        media_pert = parametros["media"]
        desvio_pert = parametros["desvio"]
        q = (np.arange(n_quantis, dtype=float) + 0.5) / n_quantis
        quantis_pert = (
            norm.ppf(q, loc=media_pert, scale=desvio_pert)
            if desvio_pert > 0
            else np.full(n_quantis, media_pert)
        )

        linhas_duracoes = []
        linhas_caminhos = []
        caminho_pert = " -> ".join(parametros["caminho_critico"])

        for dist in distribuicoes:
            amostras = np.asarray(
                self._resultados_cache[dist]["duracao_projeto"], dtype=float
            )
            quantis_simulados = np.quantile(amostras, q)
            if np.std(quantis_simulados) > 0 and np.std(quantis_pert) > 0:
                correlacao_quantis = float(
                    np.corrcoef(quantis_simulados, quantis_pert)[0, 1]
                )
            else:
                correlacao_quantis = np.nan

            wasserstein_quantis = float(
                np.mean(np.abs(quantis_simulados - quantis_pert))
            )

            if desvio_pert > 0:
                ks = kstest(
                    amostras,
                    norm.cdf,
                    args=(media_pert, desvio_pert),
                )
                ks_statistica = float(ks.statistic)
                ks_pvalor = float(ks.pvalue)
            else:
                abaixo = np.mean(amostras < media_pert)
                acima = np.mean(amostras > media_pert)
                ks_statistica = float(max(abaixo, acima))
                ks_pvalor = np.nan

            if desvio_pert > 0:
                inferior = min(
                    float(np.min(amostras)), media_pert - 4.0 * desvio_pert
                )
                superior = max(
                    float(np.max(amostras)), media_pert + 4.0 * desvio_pert
                )
            else:
                inferior = min(float(np.min(amostras)), media_pert)
                superior = max(float(np.max(amostras)), media_pert)
            if superior <= inferior:
                margem = max(abs(inferior) * 1e-6, 1e-6)
                inferior -= margem
                superior += margem

            limites = np.linspace(inferior, superior, n_bins + 1)
            limites[0] = -np.inf
            limites[-1] = np.inf
            contagens_sim, _ = np.histogram(amostras, bins=limites)
            prob_sim = (contagens_sim + 0.5) / (
                len(amostras) + 0.5 * len(contagens_sim)
            )

            if desvio_pert > 0:
                prob_pert = np.diff(
                    norm.cdf(limites, loc=media_pert, scale=desvio_pert)
                )
            else:
                prob_pert = np.zeros(len(limites) - 1, dtype=float)
                indice = np.searchsorted(limites, media_pert, side="right") - 1
                prob_pert[min(max(indice, 0), len(prob_pert) - 1)] = 1.0
            prob_pert = np.clip(prob_pert, np.finfo(float).tiny, None)
            prob_pert /= np.sum(prob_pert)

            kl_sim_pert = float(
                np.sum(prob_sim * np.log(prob_sim / prob_pert))
            )
            kl_pert_sim = float(
                np.sum(prob_pert * np.log(prob_pert / prob_sim))
            )
            js_divergencia = float(
                jensenshannon(prob_sim, prob_pert, base=np.e) ** 2
            )

            linhas_duracoes.append(
                {
                    "Distribuição": dist,
                    "Média simulada (dias)": float(np.mean(amostras)),
                    "DP simulado (dias)": float(np.std(amostras)),
                    "Média PERT tradicional (dias)": media_pert,
                    "DP PERT tradicional (dias)": desvio_pert,
                    "R quantis": correlacao_quantis,
                    "KS (D)": ks_statistica,
                    "KS (p-valor)": ks_pvalor,
                    "Wasserstein-1 (dias)": wasserstein_quantis,
                    "KL(sim || PERT) (nats)": kl_sim_pert,
                    "KL(PERT || sim) (nats)": kl_pert_sim,
                    "Jensen-Shannon (nats)": js_divergencia,
                }
            )

            df_caminhos = self._resultados_cache[dist]["df_caminhos"]
            criticidades = {}
            for _, linha in df_caminhos.iterrows():
                caminho = linha["Caminho"].partition("(")[2].rsplit(")", 1)[0]
                criticidades[caminho] = float(linha["Criticidade (%)"])

            caminhos = set(criticidades)
            caminhos.add(caminho_pert)
            prob_pert_caminho = {
                caminho: 100.0 if caminho == caminho_pert else 0.0
                for caminho in caminhos
            }
            tv_caminhos = 0.5 * sum(
                abs(
                    criticidades.get(caminho, 0.0)
                    - prob_pert_caminho[caminho]
                )
                for caminho in caminhos
            )
            caminho_lider = max(
                criticidades, key=criticidades.get, default=""
            )

            for caminho in sorted(caminhos):
                prob_simulada = criticidades.get(caminho, 0.0)
                linhas_caminhos.append(
                    {
                        "Distribuição": dist,
                        "Caminho": caminho or "(sem atividades)",
                        "Criticidade Monte Carlo (%)": prob_simulada,
                        "PERT tradicional (%)": prob_pert_caminho[caminho],
                        "Diferença absoluta (p.p.)": abs(
                            prob_simulada - prob_pert_caminho[caminho]
                        ),
                        "Distância TV dos caminhos (%)": tv_caminhos,
                        "Líder coincide com PERT": caminho_lider
                        == caminho_pert,
                    }
                )

        return {
            "resumo": df_resumo,
            "duracoes": pd.DataFrame(linhas_duracoes),
            "caminhos": pd.DataFrame(linhas_caminhos),
        }

    def pert_tradicional(self):
        """PERT clássico (Malcolm et al., 1959), sem simulação.

        Cada atividade tem TE = (O + 4M + P)/6 e σ² = ((P - O)/6)². O caminho
        crítico é o de maior soma de TE (empate: maior variância) e a duração
        do projeto é aproximada por Normal(Σ TE, Σ σ²) nesse caminho.
        Correlações são ignoradas; condições definidas por dados são
        avaliadas com o ES determinístico.
        """
        self._validar_rede()
        self._preparar_nos_virtuais()

        te = {"Start": 0.0, "End": 0.0}
        var = {"Start": 0.0, "End": 0.0}
        ef = {}
        for no in nx.topological_sort(self.grafo):
            es = max(
                (ef[p] for p in self.grafo.predecessors(no)), default=0.0
            )
            if no not in ("Start", "End"):
                atv = self.atividades[no]
                if no in self.condicoes:
                    c = self.condicoes[no]
                    O, M, P = (
                        c["cenario_ate"]
                        if es <= c["limite_es"]
                        else c["cenario_acima"]
                    )
                elif None in (atv["O"], atv["M"], atv["P"]):
                    raise ValueError(
                        f"PERT tradicional não avalia a regra (função) da "
                        f"atividade '{no}'; use adicionar_condicao."
                    )
                else:
                    O, M, P = atv["O"], atv["M"], atv["P"]
                te[no] = (O + 4.0 * M + P) / 6.0
                var[no] = ((P - O) / 6.0) ** 2
            ef[no] = es + te[no]

        caminhos = list(
            nx.all_simple_paths(self.grafo, source="Start", target="End")
        )
        duracao_max = max(sum(te[n] for n in c) for c in caminhos)
        criticos = [
            c for c in caminhos
            if abs(sum(te[n] for n in c) - duracao_max) < 1e-9
        ]
        caminho = max(criticos, key=lambda c: sum(var[n] for n in c))
        media = sum(te[n] for n in caminho)
        desvio = float(np.sqrt(sum(var[n] for n in caminho)))

        self.resultado_pert_tradicional = {
            "media": media,
            "desvio": desvio,
            "caminho_critico": [
                n for n in caminho if n not in ("Start", "End")
            ],
        }

        resumo_lista = [
            {"Métrica / Percentil": "Média do Projeto",
             "Valor (Dias)": round(media, 2)},
            {"Métrica / Percentil": "Desvio Padrão",
             "Valor (Dias)": round(desvio, 2)},
        ]
        for p in self.PERCENTIS:
            resumo_lista.append(
                {
                    "Métrica / Percentil": f"Percentil {p} (P{p})",
                    "Valor (Dias)": round(
                        float(norm.ppf(p / 100.0, loc=media, scale=desvio))
                        if desvio > 0
                        else media,
                        2,
                    ),
                }
            )
        return pd.DataFrame(resumo_lista)

    # =========================================================================
    # ANÁLISES PARA O ARTIGO (ROBUSTEZ, CONVERGÊNCIA, TESTES, EXPORTAÇÃO)
    # =========================================================================
    def testes_pareados_ks(
        self, distribuicoes_teste=None, alpha=0.05, n_simulacoes=50000, seed=42
    ):
        """Teste KS de duas amostras entre cada par de distribuições.

        Compara as durações simuladas do projeto (não contra o PERT tradicional,
        como em ``comparar_com_pert_tradicional``, mas entre si). Os p-valores são
        corrigidos por Holm-Bonferroni para controlar o erro por múltiplas
        comparações (10 pares com as 5 distribuições padrão).
        """
        if distribuicoes_teste is None:
            distribuicoes_teste = self.DISTRIBUICOES
        distribuicoes = [
            self._nome_distribuicao(d) for d in distribuicoes_teste
        ]
        faltando = [d for d in distribuicoes if d not in self._resultados_cache]
        if faltando:
            self.simular_todas(
                n_simulacoes=n_simulacoes,
                seed=seed,
                distribuicoes_teste=distribuicoes,
            )

        pares = []
        p_valores = []
        for i, d1 in enumerate(distribuicoes):
            for d2 in distribuicoes[i + 1:]:
                amostras1 = self._resultados_cache[d1]["duracao_projeto"]
                amostras2 = self._resultados_cache[d2]["duracao_projeto"]
                resultado = ks_2samp(amostras1, amostras2)
                pares.append((d1, d2, float(resultado.statistic)))
                p_valores.append(float(resultado.pvalue))

        # Holm-Bonferroni: ordena p-valores, multiplica pelo número de testes
        # restantes e impõe monotonicidade (cada ajustado >= o anterior).
        m = len(p_valores)
        ordem = np.argsort(p_valores)
        p_ajustados = np.empty(m)
        maior_ate_agora = 0.0
        for rank, idx in enumerate(ordem):
            ajustado = min((m - rank) * p_valores[idx], 1.0)
            maior_ate_agora = max(maior_ate_agora, ajustado)
            p_ajustados[idx] = maior_ate_agora

        linhas = [
            {
                "Distribuição 1": d1,
                "Distribuição 2": d2,
                "KS (D)": round(ks_d, 4),
                "p-valor": round(p_valores[i], 4),
                "p-ajustado (Holm)": round(float(p_ajustados[i]), 4),
                "Significativo": bool(p_ajustados[i] < alpha),
            }
            for i, (d1, d2, ks_d) in enumerate(pares)
        ]
        return pd.DataFrame(linhas)

    def analisar_robustez(
        self,
        n_replicas=30,
        n_simulacoes=10000,
        seed_base=42,
        distribuicoes_teste=None,
    ):
        """Variabilidade das métricas entre réplicas independentes (seeds).

        Roda ``simular_todas`` ``n_replicas`` vezes com seeds diferentes e agrega
        cada (distribuição, métrica) com média, desvio padrão entre réplicas e um
        IC95% empírico (percentil 2,5/97,5 das réplicas).
        """
        if distribuicoes_teste is None:
            distribuicoes_teste = self.DISTRIBUICOES
        distribuicoes = [
            self._nome_distribuicao(d) for d in distribuicoes_teste
        ]

        valores = {}
        for i in range(n_replicas):
            df_comp = self.simular_todas(
                n_simulacoes=n_simulacoes,
                seed=seed_base + i,
                distribuicoes_teste=distribuicoes,
            )
            for _, linha in df_comp.iterrows():
                metrica = linha["Métrica / Percentil"]
                for dist in distribuicoes:
                    valores.setdefault((dist, metrica), []).append(
                        float(linha[dist])
                    )

        linhas = []
        for (dist, metrica), amostras in valores.items():
            amostras = np.asarray(amostras, dtype=float)
            ic_inf, ic_sup = np.percentile(amostras, [2.5, 97.5])
            linhas.append(
                {
                    "Distribuição": dist,
                    "Métrica / Percentil": metrica,
                    "Média (réplicas)": round(float(np.mean(amostras)), 3),
                    "DP (réplicas)": round(float(np.std(amostras)), 3),
                    "IC95% inf": round(float(ic_inf), 3),
                    "IC95% sup": round(float(ic_sup), 3),
                    "N réplicas": n_replicas,
                    "N simulações/réplica": n_simulacoes,
                }
            )
        return pd.DataFrame(linhas)

    def analisar_convergencia(
        self,
        distrib="Beta",
        n_simulacoes_max=200000,
        seed=42,
        checkpoints=None,
        percentis_foco=(50, 90, 95),
    ):
        """Estabilidade das estimativas conforme N (nº de simulações) cresce.

        Simula uma única vez com ``n_simulacoes_max`` e recalcula média/percentis
        usando prefixos crescentes da mesma amostra (válido pois as amostras são
        i.i.d.) — prática padrão para verificar convergência de Monte Carlo sem
        precisar resimular do zero em cada checkpoint.
        """
        if checkpoints is None:
            checkpoints = [
                n for n in (1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000)
                if n <= n_simulacoes_max
            ]
            if checkpoints[-1] != n_simulacoes_max:
                checkpoints.append(n_simulacoes_max)

        duracao_projeto, _, _, _ = self.simular(
            n_simulacoes=n_simulacoes_max, seed=seed, distrib_override=distrib
        )

        linhas = []
        for n in checkpoints:
            prefixo = duracao_projeto[:n]
            linha = {"N": n, "Média": float(np.mean(prefixo))}
            for p in percentis_foco:
                linha[f"P{p}"] = float(np.percentile(prefixo, p))
            linhas.append(linha)
        return pd.DataFrame(linhas)

    @classmethod
    def gerar_sintetico(
        cls,
        n_atividades=6,
        assimetria=0.5,
        seed=0,
        nome_projeto=None,
        incluir_correlacao=False,
        incluir_condicao=False,
    ):
        """Projeto-teste sintético em camadas, parametrizado pela assimetria O-M-P.

        ``assimetria`` em [0, 1]: 0 ≈ simétrico (M a meio caminho entre O e P);
        1 ≈ fortemente assimétrico à direita (M próximo de O, P bem mais distante),
        padrão típico de risco em durações reais de atividades. Serve para estudar
        sistematicamente como a divergência entre distribuições (Beta, Lognormal,
        Weibull, Normal...) varia com o grau de assimetria, generalizando as
        conclusões além dos exemplos manuais do notebook.

        ``incluir_correlacao``: sorteia correlações (±0,5) entre ~1 a cada 4
        pares de atividades, como no Projeto 1 (correlação B-G).
        ``incluir_condicao``: escolhe uma atividade com predecessoras e a
        torna condicional ao ES (como a atividade F do Projeto 1), via
        ``adicionar_condicao``.
        """
        rng = np.random.default_rng(seed)
        if nome_projeto is None:
            nome_projeto = f"Sintético (n={n_atividades}, assimetria={assimetria:g})"
        engine = cls(nome_projeto=nome_projeto)

        n_estagios = max(2, round(np.sqrt(n_atividades)))
        atividades_por_estagio = []
        restantes = n_atividades
        for i in range(n_estagios):
            estagios_restantes = n_estagios - i
            tamanho = max(1, round(restantes / estagios_restantes))
            tamanho = min(tamanho, max(1, restantes - (estagios_restantes - 1)))
            atividades_por_estagio.append(tamanho)
            restantes -= tamanho

        letras = [chr(ord("A") + i) for i in range(n_atividades)]
        nomes_estagios = []
        idx_letra = 0
        for tamanho in atividades_por_estagio:
            nomes_estagios.append(letras[idx_letra:idx_letra + tamanho])
            idx_letra += tamanho

        k = 0.5 - 0.4 * float(np.clip(assimetria, 0.0, 1.0))
        anterior = []
        for nomes in nomes_estagios:
            for nome in nomes:
                spread = float(rng.uniform(6.0, 14.0))
                o_base = float(rng.uniform(1.0, 3.0))
                O = o_base
                M = O + k * spread
                P = O + spread
                if anterior:
                    n_preds = rng.integers(1, len(anterior) + 1)
                    preds = list(
                        rng.choice(anterior, size=n_preds, replace=False)
                    )
                else:
                    preds = []
                engine.adicionar_atividade(
                    nome, O=O, M=M, P=P, predecessoras=preds
                )
            anterior = nomes

        if incluir_correlacao:
            nomes_todos = list(engine.atividades)
            pares = [
                (a, b)
                for i, a in enumerate(nomes_todos)
                for b in nomes_todos[i + 1:]
            ]
            if pares:
                n_pares = max(1, round(len(nomes_todos) / 4))
                idx = rng.choice(
                    len(pares), size=min(n_pares, len(pares)), replace=False
                )
                df_corr = pd.DataFrame(
                    np.eye(len(nomes_todos)),
                    index=nomes_todos,
                    columns=nomes_todos,
                )
                for i in np.atleast_1d(idx):
                    a, b = pares[i]
                    rho = float(rng.uniform(-0.5, 0.5))
                    df_corr.loc[a, b] = df_corr.loc[b, a] = rho
                engine.definir_matriz_correlacao(df_corr)

        if incluir_condicao:
            candidatas = [n for estagio in nomes_estagios[1:] for n in estagio]
            if candidatas:
                escolhida = str(rng.choice(candidatas))
                dados = engine.atividades[escolhida]
                O, M, P = dados["O"], dados["M"], dados["P"]
                limite_es = float(rng.uniform(5.0, 15.0))
                fator = float(rng.uniform(1.3, 1.8))
                engine.adicionar_condicao(
                    escolhida,
                    limite_es=limite_es,
                    cenario_ate=(O, M, P),
                    cenario_acima=(O, M * fator, P * fator),
                )

        return engine

    def exportar_para_interface(self, nome_arquivo=None, pasta="projetos_teste"):
        """Salva o projeto em ``pasta`` no formato lido por ``interface_pert.py``
        (dropdown "Salvos" do widget): ``<nome_arquivo>_atividades.csv`` sempre,
        e ``_correlacoes.csv``/``_condicoes.csv`` quando o projeto tiver
        correlação (``definir_matriz_correlacao``) ou condição (``adicionar_condicao``).
        """
        if nome_arquivo is None:
            nome_arquivo = re.sub(
                r"[^a-z0-9]+", "_", self.nome_projeto.lower()
            ).strip("_")

        linhas = []
        for nome, dados in self.atividades.items():
            linhas.append(
                {
                    "Atividade": nome,
                    "Descricao": dados.get("descricao", ""),
                    "O": dados["O"],
                    "M": dados["M"],
                    "P": dados["P"],
                    "Predecessoras": ";".join(dados["predecessoras"]),
                }
            )

        pasta_destino = Path(pasta)
        pasta_destino.mkdir(parents=True, exist_ok=True)
        caminho = pasta_destino / f"{nome_arquivo}_atividades.csv"
        pd.DataFrame(linhas).to_csv(caminho, index=False, encoding="utf-8")

        if self.matriz_correlacao is not None:
            cols = list(self.matriz_correlacao.columns)
            linhas_corr = []
            for i, a1 in enumerate(cols):
                for a2 in cols[i + 1:]:
                    rho = float(self.matriz_correlacao.loc[a1, a2])
                    if rho != 0:
                        linhas_corr.append(
                            {"Atividade_1": a1, "Atividade_2": a2, "Correlacao": rho}
                        )
            if linhas_corr:
                pd.DataFrame(linhas_corr).to_csv(
                    pasta_destino / f"{nome_arquivo}_correlacoes.csv",
                    index=False,
                    encoding="utf-8",
                )

        if self.condicoes:
            linhas_cond = [
                {
                    "Atividade": nome,
                    "Limite_ES": c["limite_es"],
                    "O_ate": c["cenario_ate"][0],
                    "M_ate": c["cenario_ate"][1],
                    "P_ate": c["cenario_ate"][2],
                    "O_acima": c["cenario_acima"][0],
                    "M_acima": c["cenario_acima"][1],
                    "P_acima": c["cenario_acima"][2],
                }
                for nome, c in self.condicoes.items()
            ]
            pd.DataFrame(linhas_cond).to_csv(
                pasta_destino / f"{nome_arquivo}_condicoes.csv",
                index=False,
                encoding="utf-8",
            )

        return caminho

    def exportar_resultados(
        self,
        diretorio="resultados_artigo",
        prefixo=None,
        n_simulacoes=50000,
        seed=42,
        incluir_testes_ks=True,
    ):
        """Salva tabelas (CSV) e figuras (PNG 300dpi + PDF vetorial) para o artigo.

        Reaproveita ``comparar_com_pert_tradicional``, ``gerar_tabela_premissas``,
        ``testes_pareados_ks`` e os métodos ``plotar_*`` existentes — nenhuma
        lógica de simulação ou plot é duplicada aqui, só orquestração de saída.
        """
        if prefixo is None:
            prefixo = re.sub(r"[^a-z0-9]+", "_", self.nome_projeto.lower()).strip("_")

        pasta = Path(diretorio)
        pasta.mkdir(parents=True, exist_ok=True)

        df_premissas = self.gerar_tabela_premissas()
        df_premissas.to_csv(pasta / f"{prefixo}_premissas.csv", index=False)

        comparacao = self.comparar_com_pert_tradicional(
            n_simulacoes=n_simulacoes, seed=seed
        )
        comparacao["resumo"].to_csv(
            pasta / f"{prefixo}_resumo_comparativo.csv", index=False
        )
        comparacao["duracoes"].to_csv(
            pasta / f"{prefixo}_duracoes_comparacao.csv", index=False
        )
        comparacao["caminhos"].to_csv(
            pasta / f"{prefixo}_caminhos_comparacao.csv", index=False
        )

        resultado = {"premissas": df_premissas, **comparacao}

        if incluir_testes_ks:
            df_ks = self.testes_pareados_ks(
                n_simulacoes=n_simulacoes, seed=seed
            )
            df_ks.to_csv(pasta / f"{prefixo}_testes_ks_pareados.csv", index=False)
            resultado["testes_ks"] = df_ks

        figuras = {
            "grafo": self.plotar_grafo(mostrar=False),
            "curva_s": self.plotar_curva_s_comparativa(mostrar=False),
            "histograma": self.plotar_histograma_comparativo(mostrar=False),
        }
        for nome_fig, fig in figuras.items():
            for ext in ("png", "pdf"):
                caminho = pasta / f"{prefixo}_{nome_fig}.{ext}"
                fig.savefig(caminho, dpi=300, bbox_inches="tight")
            plt.close(fig)

        return resultado

    # =========================================================================
    # GRÁFICOS
    # =========================================================================
    def plotar_grafo(self, figsize=(10, 5), mostrar=True):
        """Desenha a rede. Retorna a figura (útil em plataformas web)."""
        self._preparar_nos_virtuais()
        G = self.grafo
        generations = list(nx.topological_generations(G))
        pos = {}
        for x_idx, layer in enumerate(generations):
            n_nodes = len(layer)
            for y_idx, node in enumerate(layer):
                pos[node] = (x_idx, y_idx - (n_nodes - 1) / 2.0)

        fig = plt.figure(figsize=figsize)
        node_colors = [
            "#6c757d" if node in ("Start", "End") else "#1f77b4"
            for node in G.nodes
        ]

        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=2000, alpha=0.95
        )
        nx.draw_networkx_labels(
            G,
            pos,
            font_color="white",
            font_weight="bold",
            font_size=11,
        )
        nx.draw_networkx_edges(
            G,
            pos,
            arrowstyle="-|>",
            arrowsize=20,
            edge_color="#495057",
            width=2,
            node_size=2000,
        )
        plt.axis("off")
        plt.tight_layout()
        if mostrar:
            plt.show()
        return fig

    def plotar_curva_s_comparativa(self, figsize=(11, 6), mostrar=True):
        """Curva S por distribuição. Retorna a figura."""
        if not self._resultados_cache:
            self.simular_todas()

        fig = plt.figure(figsize=figsize)

        for dist_nome, res in self._resultados_cache.items():
            dur = np.sort(res["duracao_projeto"])
            p_acumulada = np.linspace(0, 100, len(dur))
            cor, ls = self.ESTILOS.get(dist_nome, (None, "-"))
            plt.plot(
                dur,
                p_acumulada,
                label=dist_nome,
                linewidth=2.5,
                color=cor,
                linestyle=ls,
            )

        trad = self.resultado_pert_tradicional
        if trad and trad["desvio"] > 0:
            x_min, x_max = plt.xlim()
            x = np.linspace(x_min, x_max, 400)
            cor, ls = self.ESTILOS["PERT_Tradicional"]
            plt.plot(
                x,
                norm.cdf(x, loc=trad["media"], scale=trad["desvio"]) * 100,
                label="PERT_Tradicional (Normal)",
                linewidth=2.0,
                color=cor,
                linestyle=ls,
            )

        plt.axhline(
            y=95,
            color="#d62728",
            linestyle="--",
            linewidth=2.0,
            alpha=0.85,
            label="Nível 95% (P95)",
        )

        plt.title(
            f"Curva S Comparativa (CDF) - {self.nome_projeto}",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Duração do Projeto (Dias)", fontsize=12)
        plt.ylabel("Probabilidade Acumulada (%)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.ylim(-2, 105)

        plt.legend(
            title="Distribuição",
            fontsize=12,
            title_fontsize=13,
            loc="lower right",
            frameon=True,
            facecolor="white",
            framealpha=0.9,
        )

        plt.tight_layout()
        if mostrar:
            plt.show()
        return fig

    def plotar_histograma_comparativo(
        self, figsize=(11, 5.5), bins=60, mostrar=True
    ):
        """Histograma (densidade) da duração do projeto por distribuição."""
        if not self.resultados:
            self.simular_todas()

        fig = plt.figure(figsize=figsize)
        for dist_nome, dur in self.resultados.items():
            cor, ls = self.ESTILOS.get(dist_nome, (None, "-"))
            plt.hist(
                dur, bins=bins, density=True, color=cor, alpha=0.12
            )
            dur = np.asarray(dur, dtype=float)
            if np.ptp(dur) > 0:
                x = np.linspace(dur.min(), dur.max(), 400)
                plt.plot(
                    x,
                    gaussian_kde(dur)(x),
                    linewidth=2.5,
                    linestyle=ls,
                    color=cor,
                    label=dist_nome,
                )

        trad = self.resultado_pert_tradicional
        if trad is None:
            self.pert_tradicional()
            trad = self.resultado_pert_tradicional
        if trad and trad["desvio"] > 0:
            x_min, x_max = plt.xlim()
            x_min = min(x_min, trad["media"] - 4.0 * trad["desvio"])
            x_max = max(x_max, trad["media"] + 4.0 * trad["desvio"])
            x = np.linspace(x_min, x_max, 400)
            cor, ls = self.ESTILOS["PERT_Tradicional"]
            plt.plot(
                x,
                norm.pdf(x, loc=trad["media"], scale=trad["desvio"]),
                label="PERT tradicional (Normal)",
                linewidth=2.0,
                color=cor,
                linestyle=ls,
            )

        plt.title(
            f"Distribuição da Duração do Projeto - {self.nome_projeto}",
            fontsize=13,
            fontweight="bold",
        )
        plt.xlabel("Duração do Projeto (Dias)", fontsize=11)
        plt.ylabel("Densidade de Probabilidade", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(title="Distribuição", frameon=True)
        plt.tight_layout()
        if mostrar:
            plt.show()
        return fig

    def plotar_convergencia(
        self, df_convergencia, distrib, figsize=(9, 5), mostrar=True
    ):
        """Plota as curvas de convergência geradas por ``analisar_convergencia``."""
        fig = plt.figure(figsize=figsize)
        colunas_metricas = [c for c in df_convergencia.columns if c != "N"]
        for metrica in colunas_metricas:
            plt.plot(
                df_convergencia["N"],
                df_convergencia[metrica],
                marker="o",
                linewidth=2.0,
                label=metrica,
            )
        plt.xscale("log")
        plt.title(
            f"Convergência de Monte Carlo - {distrib} ({self.nome_projeto})",
            fontsize=13,
            fontweight="bold",
        )
        plt.xlabel("N (nº de simulações)", fontsize=11)
        plt.ylabel("Duração do Projeto (Dias)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(title="Métrica", frameon=True)
        plt.tight_layout()
        if mostrar:
            plt.show()
        return fig
