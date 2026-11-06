# Comparação das distribuições de duração e da criticidade dos caminhos

## PERT tradicional e distribuição Beta analítica ancorada na moda

O PERT tradicional representa cada atividade por três estimativas: otimista
\(O\), mais provável \(M\) e pessimista \(P\). A duração esperada é

\[
TE = \frac{O + 4M + P}{6},
\]

e o desvio-padrão aproximado é

\[
\sigma_{\mathrm{PERT}} = \frac{P-O}{6}.
\]

Na aproximação clássica da duração do projeto, identificam-se os tempos
esperados das atividades, determina-se um caminho crítico determinístico e
aproximam-se a média e a variância do prazo total pela soma dos valores desse
caminho. A distribuição do prazo é então aproximada por uma Normal. Essa
abordagem é uma referência analítica útil, mas não representa explicitamente a
mudança do caminho crítico entre realizações, as correlações entre atividades
ou toda a variabilidade dos caminhos quase críticos (Malcolm et al., 1959).

Neste estudo, a **distribuição Beta analítica ancorada na moda** é uma
distribuição Beta contínua definida para cada atividade no intervalo \([O,P]\),
com moda \(M\). O código adota a parametrização correspondente a \(\lambda=4\),
com parâmetros

\[
\alpha = 1 + 4\frac{M-O}{P-O},
\qquad
\beta = 1 + 4\frac{P-M}{P-O}.
\]

Sua média coincide com \(TE\), mas sua variância é a variância própria da Beta,

\[
\operatorname{Var}(X) = \frac{(TE-O)(P-TE)}{7},
\]

e não necessariamente \(((P-O)/6)^2\), que é a aproximação de desvio-padrão
usada pelo PERT tradicional. As durações totais do projeto sob a distribuição
Beta são obtidas por simulação da rede; como o prazo corresponde ao maior tempo
entre caminhos, sua distribuição não é, em geral, uma Beta. Portanto, a
**Beta analítica ancorada na moda** e o **PERT tradicional com aproximação
Normal do prazo** são modelos distintos, e a comparação entre eles é pertinente
(Clark, 1961; Keefer; Verdini, 1993).

## Distribuição simulada da duração do projeto

Para cada distribuição considerada, sejam \(T_1,\ldots,T_N\) os prazos
simulados e seja \(F_0\) a distribuição Normal do prazo fornecida pelo PERT
tradicional. A comparação deve considerar tanto medidas de posição e
dispersão quanto a forma da distribuição e os quantis do prazo. Nenhuma medida
isolada demonstra que duas distribuições são equivalentes.

O coeficiente \(R\) apresentado neste trabalho é a correlação de Pearson entre
os quantis empíricos dos prazos simulados e os quantis da Normal do PERT,
avaliados nos mesmos níveis de probabilidade. Ele indica semelhança de
ordenação/forma, mas não é uma medida de erro: uma correlação alta pode ocorrer
mesmo quando há diferença sistemática de localização ou escala. Por isso, deve
ser interpretado junto com diferenças entre médias, desvios e quantis e com
medidas de distância.

A estatística de Kolmogorov-Smirnov é

\[
D_N = \sup_t \left|F_N(t)-F_0(t)\right|,
\]

em que \(F_N\) é a distribuição acumulada empírica dos prazos simulados. Valores
menores indicam maior proximidade entre as funções acumuladas. O código também
apresenta o p-valor do teste de uma amostra; ele não deve ser tratado como
medida de tamanho do efeito nem como prova de equivalência. Em particular,
amostras Monte Carlo grandes podem tornar diferenças pequenas estatisticamente
detectáveis (Kolmogorov, 1933; Smirnov, 1948).

A distância de Wasserstein de ordem 1 é

\[
W_1(F_N,F_0) = \int_0^1
\left|F_N^{-1}(u)-F_0^{-1}(u)\right|\,du.
\]

Ela é expressa na mesma unidade do prazo (por exemplo, dias) e resume a
diferença entre quantis. A implementação aproxima a integral por uma média em
uma grade comum de quantis (Gibbs; Su, 2002).

As divergências de Kullback-Leibler são calculadas nas duas direções,
\(D_{\mathrm{KL}}(F_N\Vert F_0)\) e \(D_{\mathrm{KL}}(F_0\Vert F_N)\), a partir
de histogramas com classes comuns. KL é assimétrica e sensível à escolha das
classes; a suavização de Jeffreys (0,5 por classe) evita logaritmos de zero,
mas não elimina essa sensibilidade. A divergência de Jensen-Shannon é
simétrica e limitada, e é apresentada como complemento. KL e Jensen-Shannon
dependem, nesta implementação, da discretização do histograma e devem ser
reportadas com essa ressalva (Kullback; Leibler, 1951; Lin, 1991).

Essas métricas comparam os modelos simulados com a **aproximação Normal do
PERT tradicional**, não com uma distribuição verdadeira observada. Sem dados
históricos de duração, o estudo permite avaliar a sensibilidade dos resultados
à modelagem, mas não afirmar qual distribuição representa melhor projetos
reais.

## Criticidade dos caminhos

Em cada iteração \(s\), o caminho crítico é o caminho de maior duração total.
Para um caminho \(j\), sua criticidade Monte Carlo é

\[
\widehat{p}_j = \frac{1}{N}\sum_{s=1}^{N}
\mathbf{1}\{j \text{ é crítico na iteração }s\}.
\]

Assim, a criticidade representa a frequência simulada com que o caminho
determina o prazo do projeto. Ela é diferente da duração média do caminho e da
criticidade determinística calculada uma única vez pelo CPM/PERT tradicional.
Essa distinção é importante em redes nas quais distribuições de duração
diferentes podem alterar a frequência dos caminhos críticos (Hajdu; Bokor,
2016).

O PERT tradicional seleciona um único caminho crítico. Para comparar essa
seleção com as simulações, ela é representada por um vetor de referência
determinístico: probabilidade 100% para o caminho selecionado e 0% para os
demais. Para cada caminho, reportam-se a criticidade simulada e a diferença
absoluta, em pontos percentuais, em relação a esse vetor. A distância de
variação total é

\[
TV = \frac{1}{2}\sum_j
\left|\widehat{p}_j-q_j\right|,
\]

em que \(q_j\) é o vetor determinístico do PERT tradicional. Essa comparação
quantifica quanto da massa de criticidade simulada está fora do caminho
selecionado pelo método tradicional; não deve ser interpretada como erro em
relação a frequências históricas desconhecidas.

A análise deve apresentar também qual caminho lidera em cada distribuição e
se esse líder coincide com o caminho tradicional. Desse modo, resultados com
prazo médio semelhante, mas diferentes probabilidades de criticidade, não são
confundidos com invariância completa da rede.

## Escopo e interpretação

As comparações devem usar a mesma rede, as mesmas estimativas de entrada e o
mesmo número de simulações. As distribuições implementadas não impõem
necessariamente as mesmas propriedades a \(O\), \(M\) e \(P\): por exemplo,
uma distribuição não limitada superiormente não trata \(P\) como máximo
absoluto. Portanto, diferenças observadas podem refletir tanto a forma da
distribuição quanto a parametrização adotada. O relatório deve explicitar essas
regras antes de atribuir diferenças exclusivamente à “forma” da distribuição.

Os projetos de exemplo permitem verificar se o código funciona em redes
distintas, incluindo uma rede com correlação/condição e outra sem essas
configurações. Essa verificação computacional não substitui validação com
projetos reais. Para sustentar uma conclusão de pouca influência da
distribuição, recomenda-se apresentar, para cada projeto, diferenças de
prazo, distâncias entre distribuições e criticidades dos caminhos, além de
verificar estabilidade com mais de uma semente e com maior número de
simulações.

## Referências

- GIBBS, A. L.; SU, F. E. On Choosing and Bounding Probability Metrics.
  *International Statistical Review*, v. 70, n. 3, p. 419–435, 2002.
  [https://doi.org/10.1111/j.1751-5823.2002.tb00178.x](https://doi.org/10.1111/j.1751-5823.2002.tb00178.x)
- HAJDU, M.; BOKOR, O. Sensitivity analysis in PERT networks: Does activity
  duration distribution matter? *Automation in Construction*, 2016.
  [https://doi.org/10.1016/j.autcon.2016.01.003](https://doi.org/10.1016/j.autcon.2016.01.003)
- KEEFER, D. L.; VERDINI, W. A. Better Estimation of PERT Activity Time
  Parameters. *Management Science*, v. 39, n. 9, p. 1086–1091, 1993.
  [https://doi.org/10.1287/mnsc.39.9.1086](https://doi.org/10.1287/mnsc.39.9.1086)
- KOLMOGOROV, A. N. Sulla determinazione empirica di una legge di
  distribuzione. *Giornale dell'Istituto Italiano degli Attuari*, v. 4,
  p. 83–91, 1933.
- KULLBACK, S.; LEIBLER, R. A. On Information and Sufficiency.
  *The Annals of Mathematical Statistics*, v. 22, n. 1, p. 79–86, 1951.
  [https://doi.org/10.1214/aoms/1177729694](https://doi.org/10.1214/aoms/1177729694)
- LIN, J. Divergence measures based on the Shannon entropy.
  *IEEE Transactions on Information Theory*, v. 37, n. 1, p. 145–151, 1991.
  [https://doi.org/10.1109/18.61115](https://doi.org/10.1109/18.61115)
- MALCOLM, D. G.; ROSEBOOM, J. H.; CLARK, C. E.; FAZAR, W. Application of a
  Technique for Research and Development Program Evaluation. *Operations
  Research*, v. 7, n. 5, p. 646–669, 1959.
  [https://doi.org/10.1287/opre.7.5.646](https://doi.org/10.1287/opre.7.5.646)
- SMIRNOV, N. Table for estimating the goodness of fit of empirical
  distributions. *The Annals of Mathematical Statistics*, v. 19, n. 2,
  p. 279–281, 1948.
- CLARK, C. E. The PERT Model for the Distribution of an Activity Time.
  *Operations Research*, v. 9, n. 3, p. 405–406, 1961.
  [https://doi.org/10.1287/opre.9.3.405](https://doi.org/10.1287/opre.9.3.405)
