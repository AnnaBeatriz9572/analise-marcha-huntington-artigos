# Estudo descritivo do artigo: identificação da Doença de Huntington por dinâmica da marcha (An Automatic Method for Identifying Huntington’s Disease using Gait Dynamics)

## Referência do artigo

FELIX, Juliana Paula; VIEIRA, Flávio Henrique Teles; VIEIRA, Gabriel da Silva; FRANCO, Ricardo Augusto Pereira; COSTA, Ronaldo Martins da; SALVINI, Rogerio Lopes. **An Automatic Method for Identifying Huntington’s Disease using Gait Dynamics**. In: **IEEE 31st International Conference on Tools with Artificial Intelligence (ICTAI)**, 2019. *Proceedings [...]*. [S. l.]: IEEE, 2019. p. 1659-1663. DOI: 10.1109/ICTAI.2019.00243.

---

## 1. Objetivo do estudo

O artigo propõe um método automático para identificar a Doença de Huntington a partir de informações da dinâmica da marcha. A ideia central é utilizar sinais temporais obtidos durante a caminhada dos sujeitos, realizar um pré-processamento desses sinais, extrair características numéricas e, por fim, aplicar classificadores de aprendizado de máquina para distinguir pacientes com Huntington de indivíduos controles saudáveis.

No notebook desenvolvido, o objetivo foi reproduzir, em Python, a lógica metodológica apresentada no artigo. Assim, o código foi organizado seguindo o pipeline descrito pelos autores:

```text
Preprocessing → Feature Extraction → Classification → Diagnosis Output
```

Neste documento, cada etapa do notebook é descrita de forma conceitual, relacionando o que foi implementado no código com o método apresentado no artigo.

---

## 2. Contextualização do problema

A Doença de Huntington é uma doença genética neurodegenerativa que afeta funções motoras, cognitivas e psiquiátricas. Como a doença interfere no controle motor, a marcha pode apresentar alterações importantes. Dessa forma, sinais relacionados ao modo como uma pessoa caminha podem carregar informações úteis para diferenciar indivíduos saudáveis de pacientes com Huntington.

O artigo parte dessa hipótese: alterações na dinâmica da marcha podem ser utilizadas como entrada para um sistema de classificação automática. Em vez de utilizar imagens ou exames clínicos complexos, o método trabalha com séries temporais associadas a parâmetros da marcha, como intervalo da passada, intervalo de balanço, intervalo de apoio e duplo apoio.

---

## 3. Base de dados utilizada

A base utilizada no artigo é a **Gait Dynamics in Neuro-Degenerative Disease Database**, disponível na plataforma PhysioNet. Essa base contém registros de marcha coletados por sensores posicionados nos calçados dos participantes.

No experimento reproduzido no notebook, foram utilizados apenas dois grupos:

- pacientes com Doença de Huntington, representados pelos arquivos iniciados por `hunt`;
- indivíduos controles saudáveis, representados pelos arquivos iniciados por `control`.

No total, foram considerados:

```text
20 registros de pacientes com Huntington
16 registros de controles saudáveis
```

Portanto, a classificação realizada é binária:

```text
1 → Huntington
0 → Controle saudável
```

---

## 4. Organização inicial dos arquivos no código

No notebook, a primeira parte do código foi responsável por localizar e organizar os arquivos da base de dados. Para isso, foi utilizada a biblioteca `pathlib`, que facilita o trabalho com caminhos de arquivos em Python.

A função `ordenar_numeros()` foi criada para ordenar os arquivos de acordo com o número presente no nome. Isso evita que arquivos como `hunt10.ts` apareçam antes de `hunt2.ts`, o que poderia acontecer em uma ordenação alfabética simples.

Em seguida, os arquivos foram separados em duas listas:

```python
hunt_data
ctrl_data
```

A lista `hunt_data` contém os arquivos dos pacientes com Huntington, enquanto `ctrl_data` contém os arquivos dos controles saudáveis. Essa separação é importante porque, posteriormente, cada grupo recebe um rótulo diferente no processo de classificação.

---

## 5. Definição das colunas da base

Cada arquivo `.ts` da base possui colunas relacionadas a diferentes medidas da marcha. No notebook, foram definidas duas listas principais:

```python
COLUNAS_COMPLETAS
COLUNAS_SINAIS
```

A lista `COLUNAS_COMPLETAS` contém todas as colunas presentes nos arquivos, incluindo o tempo decorrido e medidas em segundos e em porcentagem. Já a lista `COLUNAS_SINAIS` contém apenas os sinais que foram utilizados na extração de características.

Os sinais considerados foram:

```text
Left Stride Interval (sec)
Right Stride Interval (sec)
Left Swing Interval (sec)
Right Swing Interval (sec)
Left Stance Interval (sec)
Right Stance Interval (sec)
Double Support Interval (sec)
```

Esses sinais representam diferentes fases da marcha. O `stride interval` corresponde ao intervalo da passada; o `swing interval` corresponde ao tempo em que o pé está no ar; o `stance interval` corresponde ao tempo em que o pé permanece em contato com o chão; e o `double support interval` representa o período em que os dois pés estão em contato com o chão.

---

## 6. Carregamento dos arquivos

A função `carregar_arquivo_ts()` foi criada para ler cada arquivo `.ts` e transformá-lo em um `DataFrame` do pandas. Como os arquivos são separados por espaços, foi utilizado o parâmetro:

```python
sep=r"\s+"
```

Além disso, como os arquivos não possuem cabeçalho explícito, os nomes das colunas foram atribuídos manualmente por meio da lista `COLUNAS_COMPLETAS`.

Essa função é importante porque padroniza o carregamento dos dados. Assim, qualquer arquivo da base pode ser lido da mesma forma, evitando repetição de código.

---

## 7. Pré-processamento dos dados

O pré-processamento no artigo tem duas etapas principais:

1. remoção dos primeiros 20 segundos de caminhada;
2. aplicação de um filtro baseado na mediana para tratar valores discrepantes.

Essas duas etapas também foram implementadas no notebook.

### 7.1 Remoção dos primeiros 20 segundos

O artigo remove os primeiros 20 segundos dos registros para diminuir efeitos iniciais da caminhada, como instabilidade no início do movimento. No código, isso foi feito pela função:

```python
remover_20_segundos(df)
```

Essa função utiliza a coluna:

```text
Elapsed Time (sec)
```

E mantém apenas as linhas em que o tempo decorrido é maior que 20 segundos.

Conceitualmente, a função faz:

```text
manter apenas observações com tempo > 20 segundos
```

Essa etapa prepara as séries para que o cálculo das características seja feito sobre uma região mais estável da caminhada.

### 7.2 Filtro da mediana

Após a remoção dos primeiros 20 segundos, foi aplicado um filtro para tratar valores discrepantes. No artigo, valores considerados ruídos são aqueles que estão acima ou abaixo de três desvios-padrão em relação à mediana da série.

No notebook, isso foi implementado na função:

```python
filtro_mediana(serie)
```

Para cada série temporal, o código calcula:

```python
mediana = serie.median()
desvio_padrao = serie.std()
```

Depois define dois limites:

```python
limite_inferior = mediana - 3 * desvio_padrao
limite_superior = mediana + 3 * desvio_padrao
```

Os valores fora desse intervalo são considerados outliers e substituídos pela mediana da própria série. Assim, a série mantém o mesmo tamanho, mas os valores extremos são suavizados.

A função retorna duas informações:

```text
série filtrada
quantidade de outliers encontrados
```

Essa informação sobre a quantidade de outliers foi mantida no notebook para fins de análise e conferência.

---

## 8. Extração de características

Depois do pré-processamento, cada série temporal da marcha foi transformada em características numéricas. O artigo utiliza duas características principais:

```text
CV → Coeficiente de Variação
alpha → Fractal Scaling Index calculado por DFA
```

A ideia dessa etapa é transformar uma série temporal completa em um vetor pequeno de características. No caso do artigo, cada série é representada por apenas dois valores:

```text
[CV, alpha]
```

Esse vetor de tamanho 2 é usado como entrada para os classificadores.

### 8.1 Coeficiente de Variação

O Coeficiente de Variação mede a variabilidade relativa de uma série temporal. Ele é calculado pela fórmula:

```text
CV = 100 × (σ / μ)
```

Em que:

```text
σ = desvio-padrão da série
μ = média da série
```

No notebook, esse cálculo foi implementado na função:

```python
calcular_cv(serie)
```

Essa função calcula a média, o desvio-padrão e retorna o CV. No contexto da marcha, o CV indica o quanto um sinal varia em relação ao seu valor médio. Valores maiores de CV indicam maior variabilidade da marcha.

### 8.2 Cálculo do alpha pelo DFA

A segunda característica extraída foi o índice `alpha`, calculado pelo método DFA, isto é, **Detrended Fluctuation Analysis**. Esse índice é usado para avaliar propriedades de correlação e estrutura temporal em séries temporais.

No notebook, o cálculo foi implementado na função:

```python
calcular_alpha(serie, n_min=10, n_max=20)
```

O procedimento implementado segue a lógica descrita no artigo:

1. a série é transformada em um vetor numérico;
2. a média da série é removida;
3. a série centralizada é integrada por soma acumulada;
4. a série integrada é dividida em janelas de tamanho `n`;
5. em cada janela, uma reta é ajustada por mínimos quadrados;
6. a tendência local é removida;
7. calcula-se a flutuação RMS;
8. repete-se o processo para janelas entre 10 e 20;
9. constrói-se a relação log-log entre `F(n)` e `n`;
10. a inclinação da reta ajustada é definida como `alpha`.

No código, foram utilizadas janelas de:

```text
10 ≤ n ≤ 20
```

Essa faixa segue a configuração apresentada no artigo.

Algumas linhas da função, como verificações de tamanho da série ou número mínimo de segmentos, são proteções computacionais da implementação em Python. Elas não alteram a lógica metodológica, apenas evitam erros durante a execução.

---

## 9. Processamento completo de cada paciente

Depois de criar as funções individuais, foi definida uma função para processar cada arquivo de paciente:

```python
processar_paciente(caminho_arquivo, grupo, rotulo)
```

Essa função reúne todas as etapas anteriores:

1. carrega o arquivo `.ts`;
2. remove os primeiros 20 segundos;
3. aplica o filtro da mediana em cada sinal;
4. calcula o CV;
5. calcula o alpha;
6. armazena os resultados em uma lista de dicionários.

Para cada paciente e para cada sinal da marcha, a função guarda:

```text
arquivo
grupo
rotulo
sinal
cv
alpha
outliers
```

Essa função é importante porque automatiza o processamento de todos os arquivos. Em vez de repetir o mesmo código para cada paciente, o notebook processa todos os arquivos em laços `for`.

---

## 10. Construção da base final de características

Após definir o processamento individual, o código aplica a função a todos os pacientes com Huntington e a todos os controles saudáveis.

O resultado é reunido na tabela:

```python
df_features
```

Essa tabela é a base final usada na classificação. Cada linha representa a combinação entre um paciente e um sinal da marcha.

Como foram utilizados 36 sujeitos e 7 sinais, o tamanho esperado da tabela é:

```text
36 × 7 = 252 linhas
```

No notebook, foi verificado que a tabela final possui:

```text
252 linhas e 7 colunas
```

Também foi verificado que não existem valores ausentes nas colunas `cv` e `alpha`, o que é necessário para o treinamento dos classificadores.

---

## 11. Organização dos dados para classificação

Antes de treinar os modelos, os dados foram organizados por sinal. Isso foi feito porque o artigo avalia os classificadores separadamente para cada parâmetro da marcha.

No notebook, foi criado um dicionário chamado:

```python
dados_por_sinal
```

Para cada sinal, foram separados:

```python
X = df_sinal[["cv", "alpha"]]
y = df_sinal["rotulo"]
```

Ou seja:

```text
X → características de entrada do modelo
y → classe real do sujeito
```

No caso deste estudo:

```text
X = CV e alpha
y = 1 para Huntington, 0 para Controle
```

Cada sinal possui 36 linhas, correspondentes aos 36 sujeitos da base. Cada sujeito é representado por duas características: CV e alpha.

---

## 12. Classificação

A etapa de classificação utiliza os vetores de características para distinguir pacientes com Huntington de controles saudáveis.

No notebook, foram utilizados os cinco classificadores avaliados no artigo:

```text
SVM → Support Vector Machine
KNN → K-Nearest Neighbors
NB → Naive Bayes
LDA → Linear Discriminant Analysis
DT → Decision Tree
```

Esses modelos foram organizados em um dicionário:

```python
MODELOS = {...}
```

Essa organização não é uma etapa do artigo em si, mas uma forma prática de implementar, em Python, a comparação entre todos os classificadores.

O SVM foi configurado com kernel linear, de acordo com o artigo. O KNN foi configurado com distância euclidiana. Na implementação em Python, foi utilizado `k = 1`, escolha que deve ser descrita como uma decisão de implementação.

---

## 13. Validação Leave-One-Out

Para avaliar os modelos, foi utilizada validação **Leave-One-Out Cross-Validation**, também chamada de LOOCV.

Como cada sinal possui 36 sujeitos, o LOOCV funciona da seguinte forma:

```text
Rodada 1: treina com 35 sujeitos e testa com 1
Rodada 2: treina com 35 sujeitos e testa com outro 1
...
Rodada 36: treina com 35 sujeitos e testa com o último sujeito
```

No final, cada sujeito foi utilizado uma vez como teste.

No notebook, esse procedimento foi implementado na função:

```python
avaliar_modelo_loocv(X, y, modelo)
```

Essa função recebe as características, os rótulos e o classificador. Em seguida, aplica LOOCV, guarda as classes reais e previstas, calcula a acurácia e gera a matriz de confusão.

---

## 14. Tabela de resultados

Depois de criar a função de avaliação, o notebook executa todos os classificadores para todos os sinais.

Como foram usados 7 sinais e 5 modelos, o número esperado de resultados é:

```text
7 × 5 = 35 resultados
```

Esses resultados foram armazenados na tabela:

```python
df_resultados_classificacao
```

Cada linha dessa tabela contém:

```text
sinal
modelo
acuracia
acuracia_percentual
```

Também foi criada uma tabela comparativa de acurácia, na qual as linhas representam os modelos, as colunas representam os sinais e os valores representam a acurácia percentual.

Essa etapa permite comparar quais combinações entre sinal da marcha e classificador tiveram melhor desempenho.

---

## 15. Matriz de confusão

Após a avaliação das acurácias, foram geradas matrizes de confusão para analisar os acertos e erros dos classificadores.

A matriz de confusão permite observar:

```text
quantos pacientes HD foram classificados corretamente como HD;
quantos pacientes HD foram classificados incorretamente como controle;
quantos controles foram classificados incorretamente como HD;
quantos controles foram classificados corretamente como controle.
```

No notebook, as matrizes foram armazenadas no dicionário:

```python
matrizes_confusao
```

A leitura adotada foi:

| Classe real | Previsto HD | Previsto CO |
|---|---:|---:|
| Real HD | HD classificados corretamente | HD classificados como controle |
| Real CO | Controles classificados como HD | Controles classificados corretamente |

Essa etapa é importante porque a acurácia mostra apenas o percentual total de acertos, enquanto a matriz de confusão mostra o tipo de erro cometido pelo modelo.

---

## 16. Relação entre o artigo e a implementação em Python

O notebook reproduz a lógica central do artigo, mas algumas estruturas utilizadas são próprias da implementação em Python.

### Etapas diretamente baseadas no artigo

```text
Uso da base de marcha da PhysioNet
Seleção dos grupos Huntington e Controle
Remoção dos primeiros 20 segundos
Filtro da mediana com limite de três desvios-padrão
Extração de CV
Extração de alpha por DFA
Uso dos classificadores SVM, KNN, NB, LDA e DT
Uso de validação Leave-One-Out
Cálculo da acurácia
Análise por matriz de confusão
```

### Etapas de organização criadas no notebook

```text
Uso de dicionários para organizar dados por sinal
Uso de dicionário para armazenar modelos
Uso da função avaliar_modelo_loocv
Uso de DataFrames auxiliares para apresentar os resultados
Uso de proteções computacionais na função do DFA
```

Essas estruturas não mudam a metodologia do artigo; elas apenas tornam o código mais organizado e reutilizável.

---

## 17. Conclusão parcial

Neste estudo, foi implementado em Python o pipeline proposto no artigo para identificação da Doença de Huntington a partir da dinâmica da marcha. Inicialmente, os dados foram carregados e separados entre pacientes com Huntington e controles saudáveis. Em seguida, foi realizado o pré-processamento, composto pela remoção dos primeiros 20 segundos e pela aplicação do filtro da mediana para tratamento de valores discrepantes.

Na etapa de extração de características, foram calculados o Coeficiente de Variação e o índice alpha para cada sinal da marcha. Essas duas características formaram o vetor de entrada dos classificadores.

Na etapa de classificação, foram avaliados os modelos SVM, KNN, Naive Bayes, LDA e Árvore de Decisão, utilizando validação Leave-One-Out. A classificação foi feita separadamente para cada sinal da marcha, permitindo comparar quais sinais apresentaram maior capacidade de diferenciação entre pacientes com Huntington e controles saudáveis.

Por fim, as matrizes de confusão foram utilizadas para analisar de forma mais detalhada os erros e acertos dos classificadores. Com isso, o notebook passou a reproduzir o fluxo metodológico central do artigo e a gerar resultados interpretáveis para a continuidade da iniciação científica.

---

## Observações para trabalhos futuros

Como continuidade deste estudo, podem ser realizados os seguintes passos:

1. comparar os resultados obtidos no notebook com as tabelas apresentadas no artigo;
2. investigar diferenças entre a implementação em Python e a implementação original em MATLAB;
3. testar diferentes valores de `k` no KNN;
4. incluir métricas adicionais, como sensibilidade e especificidade;
5. aplicar o mesmo pipeline em outras doenças da base, como Parkinson e ALS.
