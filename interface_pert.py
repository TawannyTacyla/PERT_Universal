"""Interface de entrada de projetos PERT/CPM no Jupyter (ipywidgets).

Uso no notebook:
    from interface_pert import abrir_interface
    ui = abrir_interface()
    # depois de simular, a engine fica disponível em ui.engine
"""

import glob
import html
import os
import re
import unicodedata

import ipywidgets as w
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from IPython.display import display

from classe_pert_universal import PERTUniversalEngine

PASTA_PROJETOS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "projetos_teste"
)

# (chave no CSV, rótulo na tela, tipo, largura)
COLUNAS_ATIVIDADES = [
    ("Atividade", "Atividade", "texto", "90px"),
    ("Descricao", "Descrição", "texto", "200px"),
    ("O", "O", "numero", "70px"),
    ("M", "M", "numero", "70px"),
    ("P", "P", "numero", "70px"),
    ("Predecessoras", "Predecessoras", "texto", "140px"),
]
COLUNAS_CORRELACOES = [
    ("Atividade_1", "Atividade 1", "texto", "110px"),
    ("Atividade_2", "Atividade 2", "texto", "110px"),
    ("Correlacao", "Correlação (-1 a 1)", "numero", "140px"),
]
COLUNAS_CONDICOES = [
    ("Atividade", "Atividade", "texto", "90px"),
    ("Limite_ES", "Limite ES", "numero", "80px"),
    ("O_ate", "O (ES ≤)", "numero", "75px"),
    ("M_ate", "M (ES ≤)", "numero", "75px"),
    ("P_ate", "P (ES ≤)", "numero", "75px"),
    ("O_acima", "O (ES >)", "numero", "75px"),
    ("M_acima", "M (ES >)", "numero", "75px"),
    ("P_acima", "P (ES >)", "numero", "75px"),
]


class TabelaEditavel:
    """Tabela de campos de texto com linhas que podem ser incluídas e
    removidas. Números aceitam vírgula ou ponto decimal."""

    def __init__(self, colunas, titulo=""):
        self.colunas = colunas
        self.linhas = []
        self._corpo = w.VBox()
        cabecalho = w.HBox(
            [
                w.HTML(f"<b>{rotulo}</b>", layout=w.Layout(width=largura))
                for _, rotulo, _, largura in colunas
            ]
        )
        botao_linha = w.Button(
            description="+ linha", icon="plus", layout=w.Layout(width="110px")
        )
        botao_linha.on_click(lambda _: self.adicionar_linha())
        self.widget = w.VBox(
            [w.HTML(titulo), cabecalho, self._corpo, botao_linha]
        )

    @staticmethod
    def _formatar(valor):
        if valor is None:
            return ""
        if isinstance(valor, float):
            if pd.isna(valor):
                return ""
            return f"{valor:g}"
        return str(valor)

    def adicionar_linha(self, valores=None):
        valores = valores or {}
        campos = [
            w.Text(
                value=self._formatar(valores.get(chave)),
                layout=w.Layout(width=largura),
            )
            for chave, _, _, largura in self.colunas
        ]
        remover = w.Button(
            icon="times",
            tooltip="Remover linha",
            button_style="danger",
            layout=w.Layout(width="40px"),
        )
        linha = w.HBox(campos + [remover])
        remover.on_click(lambda _: self._remover_linha(linha))
        self.linhas.append((linha, campos))
        self._corpo.children = [l for l, _ in self.linhas]

    def _remover_linha(self, linha):
        self.linhas = [(l, c) for l, c in self.linhas if l is not linha]
        self._corpo.children = [l for l, _ in self.linhas]

    def limpar(self):
        self.linhas = []
        self._corpo.children = []

    def carregar(self, df):
        self.limpar()
        for _, registro in df.iterrows():
            self.adicionar_linha(registro.to_dict())

    def para_dataframe(self):
        registros = []
        for n_linha, (_, campos) in enumerate(self.linhas, start=1):
            textos = [c.value.strip() for c in campos]
            if not any(textos):
                continue
            registro = {}
            for (chave, rotulo, tipo, _), texto in zip(self.colunas, textos):
                if tipo == "numero" and texto:
                    try:
                        registro[chave] = float(texto.replace(",", "."))
                    except ValueError:
                        raise ValueError(
                            f"Valor inválido '{texto}' na coluna "
                            f"'{rotulo}', linha {n_linha}."
                        ) from None
                else:
                    registro[chave] = texto or None
            registros.append(registro)
        return pd.DataFrame(
            registros, columns=[chave for chave, _, _, _ in self.colunas]
        )


def _nome_arquivo(nome):
    sem_acento = (
        unicodedata.normalize("NFKD", nome)
        .encode("ascii", "ignore")
        .decode()
    )
    return re.sub(r"[^a-z0-9]+", "_", sem_acento.lower()).strip("_")


class InterfacePERT:
    def __init__(self, pasta_projetos=PASTA_PROJETOS):
        self.pasta = pasta_projetos
        self.engine = None

        self.nome = w.Text(
            value="Novo projeto",
            description="Projeto:",
            layout=w.Layout(width="360px"),
        )
        self.lista_projetos = w.Dropdown(
            description="Salvos:", layout=w.Layout(width="360px")
        )
        bt_carregar = w.Button(description="Carregar", icon="folder-open")
        bt_salvar = w.Button(description="Salvar", icon="save")
        bt_novo = w.Button(description="Novo", icon="file")
        bt_carregar.on_click(self._carregar)
        bt_salvar.on_click(self._salvar)
        bt_novo.on_click(self._novo)

        self.n_sim = w.BoundedIntText(
            value=50000,
            min=1000,
            max=1000000,
            step=1000,
            description="Simulações:",
            layout=w.Layout(width="220px"),
        )
        self.seed = w.IntText(
            value=42, description="Semente:", layout=w.Layout(width="180px")
        )
        bt_simular = w.Button(
            description="Simular", icon="play", button_style="success"
        )
        bt_simular.on_click(self._simular)

        self.tab_atividades = TabelaEditavel(
            COLUNAS_ATIVIDADES,
            "Predecessoras separadas por <b>;</b> (ex.: <code>B;D</code>). "
            "Duração fixa: O = M = P. Deixe O, M, P vazios só se a atividade "
            "tiver condição.",
        )
        self.tab_correlacoes = TabelaEditavel(
            COLUNAS_CORRELACOES,
            "Um par por linha. Pares não informados têm correlação 0.",
        )
        self.tab_condicoes = TabelaEditavel(
            COLUNAS_CONDICOES,
            "Se o início mais cedo (ES) da atividade for ≤ Limite ES, usa "
            "O/M/P (ES ≤); senão, usa O/M/P (ES >).",
        )
        abas_entrada = w.Tab(
            children=[
                self.tab_atividades.widget,
                self.tab_correlacoes.widget,
                self.tab_condicoes.widget,
            ]
        )
        for i, titulo in enumerate(["Atividades", "Correlações", "Condições"]):
            abas_entrada.set_title(i, titulo)

        self.mensagem = w.HTML()
        self.saida = w.Output()

        self.widget = w.VBox(
            [
                w.HTML("<h3>Simulação Monte Carlo PERT/CPM</h3>"),
                w.HBox([self.nome, bt_novo, bt_salvar]),
                w.HBox([self.lista_projetos, bt_carregar]),
                abas_entrada,
                w.HBox([self.n_sim, self.seed, bt_simular]),
                self.mensagem,
                self.saida,
            ]
        )
        self._atualizar_lista()
        self.tab_atividades.adicionar_linha()

    # ------------------------------------------------------------------
    # Arquivos
    # ------------------------------------------------------------------
    def _arquivos(self, base):
        return {
            parte: os.path.join(self.pasta, f"{base}_{parte}.csv")
            for parte in ("atividades", "correlacoes", "condicoes")
        }

    def _atualizar_lista(self):
        padrao = os.path.join(self.pasta, "*_atividades.csv")
        nomes = sorted(
            os.path.basename(p)[: -len("_atividades.csv")]
            for p in glob.glob(padrao)
        )
        self.lista_projetos.options = nomes

    def _avisar(self, texto, erro=False):
        cor = "#c0392b" if erro else "#1e7e34"
        self.mensagem.value = (
            f"<span style='color:{cor}'>{html.escape(texto)}</span>"
        )

    def _novo(self, _=None):
        self.nome.value = "Novo projeto"
        for tabela in (
            self.tab_atividades,
            self.tab_correlacoes,
            self.tab_condicoes,
        ):
            tabela.limpar()
        self.tab_atividades.adicionar_linha()
        self.saida.clear_output()
        self._avisar("")

    def _carregar(self, _=None):
        base = self.lista_projetos.value
        if not base:
            self._avisar("Nenhum projeto salvo na pasta.", erro=True)
            return
        arquivos = self._arquivos(base)
        self.nome.value = base
        self.tab_atividades.carregar(pd.read_csv(arquivos["atividades"]))
        for parte, tabela in (
            ("correlacoes", self.tab_correlacoes),
            ("condicoes", self.tab_condicoes),
        ):
            if os.path.exists(arquivos[parte]):
                tabela.carregar(pd.read_csv(arquivos[parte]))
            else:
                tabela.limpar()
        self.saida.clear_output()
        self._avisar(f"Projeto '{base}' carregado.")

    def _salvar(self, _=None):
        base = _nome_arquivo(self.nome.value)
        if not base:
            self._avisar("Informe o nome do projeto.", erro=True)
            return
        try:
            tabelas = self._ler_tabelas()
        except ValueError as erro:
            self._avisar(str(erro), erro=True)
            return
        os.makedirs(self.pasta, exist_ok=True)
        for parte, caminho in self._arquivos(base).items():
            tabelas[parte].to_csv(caminho, index=False, encoding="utf-8")
        self._atualizar_lista()
        self.lista_projetos.value = base
        self._avisar(f"Projeto salvo em {self.pasta} como '{base}_*.csv'.")

    def _ler_tabelas(self):
        return {
            "atividades": self.tab_atividades.para_dataframe(),
            "correlacoes": self.tab_correlacoes.para_dataframe(),
            "condicoes": self.tab_condicoes.para_dataframe(),
        }

    # ------------------------------------------------------------------
    # Simulação
    # ------------------------------------------------------------------
    def _simular(self, _=None):
        self.saida.clear_output()
        try:
            tabelas = self._ler_tabelas()
            if tabelas["atividades"].empty:
                raise ValueError("Cadastre pelo menos uma atividade.")
            self._avisar("Simulando...")
            engine = PERTUniversalEngine(nome_projeto=self.nome.value)
            engine.carregar_dados(
                tabelas["atividades"],
                tabelas["correlacoes"],
                tabelas["condicoes"],
            )
            comparacao = engine.comparar_com_pert_tradicional(
                n_simulacoes=self.n_sim.value,
                seed=self.seed.value,
            )
        except nx.NetworkXUnfeasible:
            self._avisar(
                "A rede tem um ciclo de precedências (ex.: A→B→A).",
                erro=True,
            )
            return
        except (ValueError, KeyError) as erro:
            self._avisar(f"Erro nos dados: {erro}", erro=True)
            return

        self.engine = engine
        self._avisar(
            f"Simulação concluída: {self.n_sim.value} iterações por "
            "distribuição. A engine está em ui.engine."
        )
        self._mostrar_resultados(engine, comparacao["resumo"], comparacao)

    def _mostrar_resultados(self, engine, df_resumo, comparacao):
        def tabela_html(df):
            return w.HTML(
                df.to_html(index=False, float_format="%.2f", na_rep="-")
            )

        def aba_tabelas(chave):
            saida = w.Output()
            with saida:
                for dist, res in engine._resultados_cache.items():
                    display(w.HTML(f"<b>{dist}</b>"))
                    display(tabela_html(res[chave]))
            return saida

        def aba_figura(fig):
            saida = w.Output()
            with saida:
                display(fig)
            plt.close(fig)
            return saida

        premissas = w.Output()
        with premissas:
            display(tabela_html(engine.gerar_tabela_premissas()))
        resumo = w.Output()
        with resumo:
            display(tabela_html(df_resumo))
        comparativo_pert = w.Output()
        with comparativo_pert:
            display(w.HTML("<b>Duração do projeto versus PERT tradicional</b>"))
            display(tabela_html(comparacao["duracoes"]))
            display(w.HTML("<b>Criticidade dos caminhos versus PERT tradicional</b>"))
            display(tabela_html(comparacao["caminhos"]))

        abas = [
            ("Premissas", premissas),
            ("Resumo", resumo),
            ("Comparação PERT", comparativo_pert),
            ("Caminhos", aba_tabelas("df_caminhos")),
            ("Atividades", aba_tabelas("df_atividades")),
            ("Rede", aba_figura(engine.plotar_grafo(mostrar=False))),
            (
                "Curva S",
                aba_figura(engine.plotar_curva_s_comparativa(mostrar=False)),
            ),
            (
                "Histograma",
                aba_figura(
                    engine.plotar_histograma_comparativo(mostrar=False)
                ),
            ),
        ]
        resultado = w.Tab(children=[saida for _, saida in abas])
        for i, (titulo, _) in enumerate(abas):
            resultado.set_title(i, titulo)
        with self.saida:
            display(resultado)


def abrir_interface(pasta_projetos=PASTA_PROJETOS):
    ui = InterfacePERT(pasta_projetos)
    display(ui.widget)
    return ui
