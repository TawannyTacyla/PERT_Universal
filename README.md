# PERT Universal — Simulação Monte Carlo PERT/CPM

Ferramenta em Python para estimar a duração de projetos com estimativas de três
pontos (otimista **O**, mais provável **M**, pessimista **P**). Compara quatro
distribuições simuladas por Monte Carlo com o **PERT tradicional** analítico.

Recursos:

- rede de precedências (grafo) com caminho crítico em cada iteração;
- quatro distribuições: **Beta** (ancorada na moda), **Triangular**,
  **Lognormal** e **Weibull**;
- **correlação** entre atividades (cópula gaussiana);
- **condições**: a duração de uma atividade depende do seu início mais cedo (ES);
- **PERT tradicional** (Malcolm et al., 1959) para comparação;
- comparação com o PERT tradicional: correlação entre quantis, KS,
  Wasserstein-1, KL/JS aproximadas por histograma e criticidade dos caminhos;
- criticidade de caminhos e de atividades, sensibilidade, percentis, Curva S e
  histograma;
- **interface de widgets** no Jupyter para cadastrar, salvar e simular projetos.

## Estrutura

```
PERT_Universal/
├── classe_pert_universal.py   # classe PERTUniversalEngine (motor de simulação)
├── interface_pert.py          # interface de entrada (ipywidgets)
├── Projetos_Teste_PERT.ipynb  # notebook com os projetos-teste
├── verificar_instalacao.py    # confere se o ambiente reproduz os resultados
├── projetos_teste/            # projetos salvos em CSV (3 arquivos por projeto)
├── requirements.txt
└── README.md
```

## Instalação

Requer **Python 3.11** (versão testada).

### Windows (PowerShell)

```powershell
cd PERT_Universal
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Se o PowerShell bloquear a ativação, rode uma vez
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` ou use o `cmd`:
`.venv\Scripts\activate.bat`.

### Linux / macOS

```bash
cd PERT_Universal
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Verificar

```bash
python verificar_instalacao.py
```

Deve terminar com `Instalação verificada: todos os resultados conferem.`

### Kernel do Jupyter

- **VS Code**: abra o notebook e escolha o interpretador `.venv` em
  *Select Kernel* (canto superior direito).
- **Jupyter Lab / Notebook**: registre o ambiente como kernel uma vez:

  ```bash
  python -m ipykernel install --user --name pert-universal --display-name "Python 3 (PERT Universal)"
  ```

  O Jupyter Lab não está no `requirements.txt`; se precisar,
  `pip install jupyterlab`.

## Uso

### 1. Notebook

Abra `Projetos_Teste_PERT.ipynb` e use **Restart + Run All**. Ele roda os dois
projetos-teste, compara as distribuições com o PERT tradicional e abre a
interface no final. Para cada projeto, exibe as métricas de comparação das
durações e a criticidade de cada caminho frente ao PERT tradicional. A curva
Normal do PERT aparece tanto na Curva S (CDF) quanto no histograma de densidade.

### 2. Interface de widgets

```python
from interface_pert import abrir_interface

ui = abrir_interface()
```

- Abas **Atividades**, **Correlações** e **Condições** com linhas editáveis
  (números aceitam vírgula ou ponto).
- **Salvar** grava o projeto em `projetos_teste/`; **Carregar** lista os salvos.
- **Simular** mostra Premissas, Resumo, Caminhos, Atividades, Rede, Curva S e
  Histograma. A engine da última simulação fica em `ui.engine`.

### 3. Código

```python
from classe_pert_universal import PERTUniversalEngine

engine = PERTUniversalEngine(nome_projeto="Meu projeto")

engine.adicionar_atividade("A", O=4, M=5, P=7)
engine.adicionar_atividade("B", O=1, M=7, P=13, predecessoras=["A"])
engine.adicionar_atividade("C", O=6, M=9, P=12, predecessoras=["A"])

# Condição: se C começar até o dia 6, usa (6, 9, 12); depois, (6, 12, 18)
engine.adicionar_condicao("C", limite_es=6, cenario_ate=(6, 9, 12),
                          cenario_acima=(6, 12, 18))

comparacao = engine.comparar_com_pert_tradicional(
    n_simulacoes=50000, seed=42
)
df_resumo = comparacao["resumo"]
print(comparacao["duracoes"])
print(comparacao["caminhos"])
print(engine.gerar_tabela_premissas())
print(df_resumo)
engine.plotar_curva_s_comparativa()
```

Ou a partir dos CSVs:

```python
engine.carregar_dados(
    "projetos_teste/projeto1_condicional_atividades.csv",
    "projetos_teste/projeto1_condicional_correlacoes.csv",
    "projetos_teste/projeto1_condicional_condicoes.csv",
)
```

## Formato dos CSVs

Cada projeto tem três arquivos em `projetos_teste/`, com o mesmo prefixo.

**`<projeto>_atividades.csv`**

| Coluna | Descrição |
|---|---|
| `Atividade` | identificador (ex.: `A`) |
| `Descricao` | opcional |
| `O`, `M`, `P` | estimativas; `O = M = P` = duração fixa; vazios só se houver condição |
| `Predecessoras` | separadas por `;` (ex.: `B;D`); vazio = início do projeto |
| `Distribuicao` | opcional, usada só em `simular()` sem `distrib_override` |

**`<projeto>_correlacoes.csv`** — `Atividade_1`, `Atividade_2`, `Correlacao`
(entre −1 e 1; pares não listados têm correlação 0).

**`<projeto>_condicoes.csv`** — `Atividade`, `Limite_ES`, `O_ate`, `M_ate`,
`P_ate`, `O_acima`, `M_acima`, `P_acima`.

## Métodos

### Distribuições simuladas

Com amplitude R = P − O e moda relativa m = (M − O)/R:

| Distribuição | Parametrização |
|---|---|
| **Beta analítica** (ancorada na moda) | α = 1 + 4m, β = 1 + 4(1 − m) em [O, P]. Moda = M e média = (O + 4M + P)/6 exatas; variância = (TE − O)(P − TE)/7 |
| **Triangular** | mínimo O, moda M, máximo P |
| **Lognormal** (deslocada) | origem em O, moda em M, σ = ln(R/(M − O))/3 limitado a [0,01; 1,2] |
| **Weibull** (3 parâmetros) | origem em O, moda em M e P como percentil 99% (`p_pessimista_weibull`) |

Atividades com O = M = P são constantes em todas as distribuições.

### PERT tradicional

Sem simulação. Para cada atividade, TE = (O + 4M + P)/6 e σ = (P − O)/6. O
caminho crítico é o de maior soma de TE (empate: maior variância), e a duração
do projeto é aproximada por uma Normal com média Σ TE e variância Σ σ² nesse
caminho. Ignora correlações e caminhos paralelos quase críticos; condições são
avaliadas com o ES determinístico.

As fórmulas vêm da hipótese de que cada atividade segue uma Beta. A média
1-4-1 coincide com a da Beta ancorada na moda, mas σ = (P − O)/6 só coincide
com o desvio dela quando m ≈ 0,15 ou m ≈ 0,85 (para m = 0,5, o desvio da Beta é
0,189 R contra 0,167 R).

### Correlação

Cópula gaussiana: normais correlacionadas pela decomposição de Cholesky
(a matriz é ajustada para positiva definida) e convertidas em quantis de cada
distribuição. A correlação vale entre os quantis; a correlação entre as
durações fica próxima, mas não idêntica.

## Principais métodos da classe

| Método | Faz |
|---|---|
| `adicionar_atividade(nome, O, M, P, predecessoras, descricao, distrib)` | cadastra atividade |
| `definir_matriz_correlacao(df)` | matriz de correlação (DataFrame com nomes no índice e colunas) |
| `adicionar_condicao(atividade, limite_es, cenario_ate, cenario_acima)` | condição por dados |
| `adicionar_regra_condicional(atividade, funcao)` | condição por função `f(es, n_sim, engine, distrib)` |
| `carregar_dados(atividades, correlacoes, condicoes)` | monta o projeto a partir de tabelas ou CSVs |
| `simular(n_simulacoes, seed, distrib_override)` | uma distribuição; devolve durações e tabelas |
| `simular_todas(n_simulacoes, seed, incluir_pert_tradicional)` | resumo comparativo |
| `pert_tradicional()` | PERT analítico |
| `analisar_criticidade_caminhos(dist)` | criticidade de cada caminho |
| `comparar_com_pert_tradicional(...)` | métricas de distribuição e comparação de criticidade |
| `gerar_tabela_premissas()` | TE, σ² e σ do PERT e σ da Beta |
| `plotar_grafo()`, `plotar_curva_s_comparativa()`, `plotar_histograma_comparativo()` | gráficos; `mostrar=False` devolve só a figura |

Nomes de distribuição aceitos: `Beta` (também `Beta_PERT`, `pert`),
`Triangular` (`triang`), `Lognormal` (`lognorm`), `Weibull` (`weibull_min`).

As métricas e sua interpretação estão descritas em
[`secao_teorica_metricas.md`](secao_teorica_metricas.md). A divergência KL e a
Jensen-Shannon são estimadas por histogramas e dependem da discretização; o
PERT tradicional fornece um caminho crítico determinístico, ao passo que a
simulação estima frequências de criticidade.

## Projetos-teste e resultados de referência

Média do projeto (dias), 50.000 simulações, semente 42:

| Projeto | Beta | Triangular | Lognormal | Weibull | PERT tradicional |
|---|---|---|---|---|---|
| `projeto1_condicional` — correlação B–G = −0,3; condição em F | 38,77 | 39,23 | 39,88 | 39,58 | 38,33 |
| `projeto2_aldo_mattos` — exemplo de Aldo Mattos | 28,00 | 28,00 | 29,45 | 29,00 | 28,00 |

## Limitações conhecidas

- **Lognormal e Weibull com M = O ou M = P** ficam praticamente constantes
  (ex.: a atividade (2, 2, 8) vira sempre 2 dias). Beta e Triangular tratam
  esses casos normalmente.
- O número de caminhos cresce rápido em redes grandes, porque a criticidade
  enumera todos os caminhos simples.
- A semente é global (`numpy.random.seed`).
