# Nanodegree Engenheiro de Machine Learning
## Projeto final
<p>Fernando Roberto Schwartzer</p>
<p>05 de janeiro de 2019</p>

## I. Definição
<Strong>Detecção do Modo de Transporte com modelos de Redes Neurais por Convolução (CNN) e modelos de Redes Neurais Recorrentes (RNN)</Strong>

### Visão geral do projeto
<p>Identificar os modos de transporte através de observações dos usuários, ou observação do ambiente, é um tópico crescente de pesquisa, com muitas aplicações no planejamento da mobilidade urbana. A detecção do modo de transporte fornece informações para o diagnóstico do uso da malha viária, da ocupação do solo, do deslocamento de cargas e, principalmente, dos deslocamentos das pessoas nas cidades.</p>
<p>O reconhecimento do modo de transporte do usuário pode ser considerado como uma tarefa de HAR (Human Activity Recognition). Seu objetivo é identificar que modo de transporte - caminhar, dirigir etc. - uma pessoa está usando.
Historicamente, os dados dos sensores para reconhecimento de atividades eram difíceis e caros de coletar, exigindo hardware personalizado. Agora, telefones inteligentes e outros dispositivos de rastreamento pessoal usados para monitoramento de saúde e fitness são baratos e onipresentes. Como tal, os dados de sensores destes dispositivos são mais baratos de coletar, mais comuns e, portanto, são uma versão mais comumente estudada do problema geral de reconhecimento de atividades.</p>
<p>Visando a obtenção de um modelo de Detecção do Modo de Transporte utilizando dados de sensores de telefones celulares, foi desenvolvido este projeto onde foram aplicadas redes neurais por convolução e redes neurais recorrentes num conjunto de dados (http://cs.unibo.it/projects/us-tm2017/download.html) desenvolvido na Universidade de Bolonha .</p>
<p>A escolha pelos modelos de redes neurais profundas se deu em virtude de estarem alcançando resultados de ponta para o reconhecimento da atividade humana. Eles são capazes de realizar o aprendizado de recursos automáticos a partir dos dados brutos do sensor e os modelos de desempenho superior se ajustam a recursos específicos do domínio criados manualmente.</p>
<p><Strong>CNN</Strong></p>
<p> Nas Redes Neurais Convolucionais (CNN), cada camada de rede atua como um filtro de detecção para a presença de características ou padrões específicos presentes nos dados originais. As primeiras camadas em um CNN detectam recursos que podem ser reconhecidos e interpretados de forma relativamente fácil. Camadas posteriores detectam recursos cada vez mais específicos. A última camada da CNN é capaz de fazer uma classificação ultra específica, combinando todos os recursos específicos detectados pelas camadas anteriores nos dados de entrada.</p>
  <p><img src= "https://github.com/fschwartzer/Udacity-Machine-Learning-Nanodegree/blob/master/Capstone/Depiction-of-CNN-Model-for-Accelerompter-Data.png" | width=500 /></p>
<p><Strong>RNN</Strong></p>
<p> A lógica das Redes Neurais Recorrentes (RNNs) é fazer uso de informações seqüenciais. Em uma rede neural tradicional, se assume que todas as entradas (e saídas) são independentes umas das outras. Mas, para muitas tarefas, não é uma boa abordagem. Querendo-se prever a próxima palavra em uma frase, é importante saber quais palavras vieram antes dela. Os RNNs são chamados de recorrentes porque executam a mesma tarefa para cada elemento de uma sequência, com a saída sendo dependente dos cálculos anteriores. Outra maneira de pensar sobre as RNNs é que elas têm uma “memória” que captura informações sobre o que foi calculado até o momento. Na teoria, os RNNs podem fazer uso de informações em sequências arbitrariamente longas, mas, na prática, limitam-se a olhar para trás apenas alguns passos.</p>
<p><img src= "https://github.com/fschwartzer/Udacity-Machine-Learning-Nanodegree/blob/master/Capstone/Depiction-of-LSTM-RNN-for-Activity-Recognition.png" | width=500 /></p>

### Descrição do problema
<p>O problema consiste na detecção do modo de transporte utilizado dada uma captura de dados instantânea por um determinado número de tipos de sensores. Enquadrado como uma tarefa de classificação de série temporal multivariada, é um problema desafiador, pois não há maneiras óbvias ou diretas de relacionar os dados do sensor registrado à atividades humanas específicas e cada sujeito pode realizar uma atividade com variação significativa, resultando em variações nos dados do sensor gravado. Foram utilizados dados registrados de sensores com as atividades correspondentes para assuntos específicos, com modelos ajustados a partir desses dados. Por fim, foi obtido um modelo otimizado que classifica a atividade de novos assuntos não vistos a partir de seus dados de sensor.</p>

### Métricas
<p>A métrica de avaliação que foi utilizada para quantificar o desempenho tanto do modelo de benchmark como dos modelos de solução apresentados foi a Acurácia.</p>

<img src= "https://github.com/fschwartzer/Udacity-Machine-Learning-Nanodegree/blob/master/Capstone/acuracia.png" />

<p>Acurácia geral com todos os quatro algoritmos de classificação do modelo de benchmark:</p>

| Algorithm | Accuracy on D1 | Accuracy on D2 | Accuracy of D3 |  
|---|:---:|:---:|:---:|
| Decision Tree (DT) | 76% | 78% | 86% |
| Random Forest (RF) | 81% | 89% | 93% |
| Support Vector Machine (SVM) | 76% | 86% | 90% |
| Neural Network (NN) | 76% | 87% | 91% | 

## II. Análise

### Exploração dos dados
<p>O conjunto de dados utilizados no projeto (http://cs.unibo.it/projects/us-tm2017/download.html) foi desenvolvido na Universidade de Bolonha com o esforço de diferentes pessoas:</p>
<ul>
  <li>Marco Di Felice • Professor Associado • email: marco.difelice3@unibo.it</li>
  <li>Luciano Bononi • Professor Associado • email: luciano.bononi@unibo.it</li>
  <li>Luca Bedogni • Professor Assistente • email: luca.bedogni4@unibo.it</li>
  <li>Vincenzo Lomonaco • Estudante de doutorado • email: vincenzo.lomonaco@unibo.it</li>
</ul>
<p>Colaboradores anteriores</p>
<ul>
  <li>Claudia Carpineti • Mestranda • e-mail: claudia.carpineti@studio.unibo.it</li>
  <li>Matteo Cappella • Aluno de mestrado • email: matteo.cappella@studio.unibo.it</li>
  <li>Simone Passaretti • Aluno de mestrado • email: simone.passaretti@studio.unibo.it</li>
</ul>
<p>A coleta de dados foi controlada por um aplicativo Android em execução no telefone dos usuários enquanto eles realizavam atividades. Esse aplicativo, por meio de uma interface gráfica simples, permitiu que os voluntários gravassem seu nome, iniciassem e interrompessem a coleta de dados e rotulassem a atividade que estava sendo executada. Foi pedido aos usuários para usar o aplicativo durante atividades específicas, como caminhar, estar em um carro, em um trem, em um ônibus ou ficar parado. As atividades foram rotuladas com estas abreviações:</p>

<p>T M = {bus, car, train, still, walking}</p> 

<ul>
<li><Strong>'bus':</Strong> Ônibus</li>
<li><Strong>'car':</Strong> Carro</li>
<li><Strong>'train':</Strong> Trem</li>
<li><Strong>'still':</Strong> Parado</li>
<li><Strong>'walking':</Strong> Caminhando</li>
</ul>

<p>O aplicativo registra cada evento do sensor com uma frequência máxima de 20 Hz. Os eventos ocorrem toda vez que um sensor detecta uma alteração nos parâmetros que está medindo, fornecendo quatro informações:</p>

<ul>
<li>o nome do sensor que acionou o evento;</li>
<li>o timestamp do evento;</li>
<li>a acurácia do evento;</li>
<li>os dados brutos do sensor que acionaram o evento.</li>
</ul>

#### Atributos

| Sensores de primeira classe de classificação | Sensores de segunda classe de classificação | Sensores de terceira classe de classificação | 
|---|---|---|
| Accelerometer | Accelerometer | Accelerometer |
| Sound | Sound | Sound |
|   | Orientation | Orientation |
|   | Linear acceleration | Linear acceleration |
|   |   | Speed |
| Gyroscope | Gyroscope | Gyroscope |
|   | Rotation vector | Rotation vector |
|   | Game rotation vector | Game rotation vector |
|   | Gyroscope uncalibrated | Gyroscope uncalibrated |
<ul>
<li><Strong>Accelerometer:</Strong> Acelerômetro</li>
<li><Strong>Sound:</Strong> Som</li>
<li><Strong>Orientation:</Strong> Orientação</li>
<li><Strong>Linear acceleration:</Strong> Aceleração Linear</li>
<li><Strong>Speed:</Strong> Velocidade</li>
<li><Strong>Gyroscope:</Strong> Giroscópio</li>
<li><Strong>Rotation vector:</Strong> Vetor de rotação</li>
<li><Strong>Game rotation vector:</Strong> Vetor de rotação para jogos</li>
<li><Strong>Gyroscope uncalibrated:</Strong> Giroscópio sem calibração</li>
</ul>

<p>Foram gerados recursos estatísticos baseados nas múltiplas leituras dos sensores. Para cada sensor, 4 recursos diferentes:</p>

<ul>
<li><Strong>'max':</Strong> Máximo valor obtido dentro da janela observada¹</li>
<li><Strong>'min':</Strong> Mínimo valor obtido dentro da janela observada¹</li>
<li><Strong>'mean':</Strong> Valor médio calculado dentro da janela observada¹</li>
<li><Strong>'std':</Strong> Desvio padrão calculado dentro da janela observada¹</li>
</ul>

<p>[1] Uma abordagem direta de preparação de dados que foi usada tanto para métodos clássicos de aprendizado de máquina quanto para redes neurais envolve dividir os dados do sinal de entrada em janelas de sinais, onde uma janela pode ter de um a alguns segundos de observação dos dados. Isso geralmente é chamado de "sliding window" (janela deslizante).</p>

<blockquote>
<p>"O reconhecimento da atividade humana visa inferir as ações de uma ou mais pessoas a partir de um conjunto de observações captadas por sensores. Normalmente, isso é feito seguindo uma abordagem de "sliding window" de comprimento fixo para a extração de recursos, onde dois parâmetros devem ser corrigidos: o tamanho da janela e o deslocamento."</p>

<p>— A Dynamic Sliding Window Approach for Activity Recognition, 2011</p>
 </blockquote>

Cada janela também está associada a uma atividade específica. Uma determinada janela de dados pode ter várias variáveis, como os eixos x, y e z de um sensor acelerômetro.

No projeto foram utilizadas as janelas de 5 segundos, equivalente a 1% dos dados brutos, e de 0,5 segundos, equivalente a 10% dos dados brutos.

### Visualização exploratória

<p>Os gráficos abaixo apresentam a contagem de dados por Modo de Transporte para as janelas de 0.5 segundos e 5 segundos:</p>

<li><strong>Janela de 0.5 segundos</strong></li>

<p><img src= "https://github.com/fschwartzer/Udacity-Machine-Learning-Nanodegree/blob/master/Capstone/Transportation Mode Using Count 0.5 seconds.png" | width=900 /></p>

<li><strong>Janela de 5 segundos</strong></li>

<p><img src= "https://github.com/fschwartzer/Udacity-Machine-Learning-Nanodegree/blob/master/Capstone/Transportation Mode Using Count 5 seconds.png" | width=900 /></p>

<p> Percebe-se que a amostra de dados para cada um dos Modos de Transporte está bem equilibrada.
  
<li><strong>Distribuição dos dados por atributos</strong></li>
A seguir são apresentados gráficos da distrubuição dos dados para cada um dos Modos de Transporte por atributo:
<p><img src= "https://github.com/fschwartzer/Udacity-Machine-Learning-Nanodegree/blob/master/Capstone/Feature Distribution - Accelerometer.png" | width=900 /></p>
<p><img src= "https://github.com/fschwartzer/Udacity-Machine-Learning-Nanodegree/blob/master/Capstone/Feature Distribution - Game Rotation Vector.png" | width=900 /></p>
<p><img src= "https://github.com/fschwartzer/Udacity-Machine-Learning-Nanodegree/blob/master/Capstone/Feature Distribution - Gyroscope Uncalibrated.png" | width=900 /></p>
<p><img src= "https://github.com/fschwartzer/Udacity-Machine-Learning-Nanodegree/blob/master/Capstone/Feature Distribution - Gyroscope.png" | width=900 /></p>
<p><img src= "https://github.com/fschwartzer/Udacity-Machine-Learning-Nanodegree/blob/master/Capstone/Feature Distribution - Linear Acceleration.png" | width=900 /></p>
<p><img src= "https://github.com/fschwartzer/Udacity-Machine-Learning-Nanodegree/blob/master/Capstone/Feature Distribution - Orientation.png" | width=900 /></p>
<p><img src= "https://github.com/fschwartzer/Udacity-Machine-Learning-Nanodegree/blob/master/Capstone/Feature Distribution - Rotation Vector.png" | width=900 /></p>
<p><img src= "https://github.com/fschwartzer/Udacity-Machine-Learning-Nanodegree/blob/master/Capstone/Feature Distribution - Sound.png" | width=900 /></p>
<p><img src= "https://github.com/fschwartzer/Udacity-Machine-Learning-Nanodegree/blob/master/Capstone/Feature Distribution - Speed.png" | width=900 /></p>
<p><img src= "https://github.com/fschwartzer/Udacity-Machine-Learning-Nanodegree/blob/master/Capstone/Feature Distribution - Time.png" | width=900 /></p>
<p>Percebe-se uma grande variação na escala dos dados, sendo necessário tratá-los para evitar que o desempenho preditivo dos algoritmos seja prejudicado.</p>
<p>Por isso, os dados foram tratados com o MinMaxScaler da biblioteca Scikit-Learn. Nesta abordagem, os dados são escalados para um intervalo fixo - geralmente de 0 a 1.
Com esse intervalo limitado, eliminam-se desvios padrão menores, o que pode suprimir o efeito de outliers.</p>
<p>O dimensionamento Min-Max é feito por meio da seguinte equação:</p>
<p><img src= "https://github.com/fschwartzer/Udacity-Machine-Learning-Nanodegree/blob/master/Capstone/CodeCogsEqn(1).gif" | width=200 /></p>
  
### Algoritmos e técnicas
<p>Neste projeto foram aplicados os modelos de redes neurais por convolução e modelos de redes neurais recorrentes.</p>
<p>Modelos de redes neurais profundas estão alcançando resultados de ponta para o reconhecimento da atividade humana. Eles são capazes de realizar o aprendizado de recursos automáticos a partir dos dados brutos do sensor e os modelos de desempenho superior se ajustam a recursos específicos do domínio criados manualmente.</p>

<blockquote>
  <p>
   “[…], Os procedimentos de extração de características e construção de modelos são freqüentemente executados simultaneamente nos modelos de aprendizagem profunda. Os recursos podem ser aprendidos automaticamente através da rede, em vez de serem projetados manualmente. Além disso, a rede neural profunda também pode extrair uma representação de alto nível na camada profunda, o que a torna mais adequada para tarefas complexas de reconhecimento de atividades.”
   </p>
<p>— Deep Learning for Sensor-based Activity Recognition: A Survey, 2018.</p>
 </blockquote>

Existem duas abordagens principais para as redes neurais que são apropriadas para a classificação de séries temporais e que demonstraram ter um bom desempenho no reconhecimento de atividades usando dados de sensores de telefones inteligentes e dispositivos de rastreamento de condicionamento físico.

Eles são modelos de redes neurais por convolução e modelos de redes neurais recorrentes.

<blockquote>
  <p>
    “Recomenda-se que a RNN e a LSTM reconheçam atividades curtas que tenham ordem natural, enquanto a CNN é melhor em inferir atividades repetitivas a longo prazo. A razão é que a RNN poderia fazer uso da relação de ordem do tempo entre as leituras do sensor, e a CNN é mais capaz de aprender recursos profundos contidos em padrões recursivos.”
    </p>
<p>— Deep Learning for Sensor-based Activity Recognition: A Survey, 2018.</p>
 </blockquote>

### Benchmark
<p>O modelo de referência foi o trabalho de detecção de modo de transporte realizado por equipe da Universidade de Bolonha, Itália:</p>

    @article {carpineti18,
    Author = {Claudia Carpineti, Vincenzo Lomonaco, Luca Bedogni, Marco Di Felice, Luciano Bononi},
    Journal = {Proc. of the 14th Workshop on Context and Activity Modeling and Recognition (IEEE COMOREA 2018)},
    Title = {Custom Dual Transportation Mode Detection by Smartphone Devices Exploiting Sensor Diversity},
    Year = {2018}
    }
    
<p>Pré-impressão disponível: https://arxiv.org/abs/1810.05596</p>
<p>Em seu trabalho, utilizaram 3 conjuntos de dados, aplicando 4 algoritmos.
Para cada conjunto, foram construídos quatro modelos com quatro algoritmos de classificação diferentes:</p>
<ul>
  <li>Decision Trees (DT)</li>
  <li>Random Forest (RF)</li>
  <li>Support Vector Machines(SVM)</li>
  <li>Neural Network (NN)</li>
</ul>

<p>Os sensores incluídos no primeiro conjunto (parâmetro 1) foram acelerômetro, som e giroscópio. Esses três sensores possuem os maiores valores de precisão obtidos individualmente.</p>
<p>O primeiro conjunto de dados é formado por doze recursos, quatro para cada sensor. Foi realizada a classificação com os quatro algoritmos de classificação mencionados anteriormente. A precisão geral dos algoritmos está entre 82% e 88%. Mesmo que a floresta aleatória produza os maiores valores de precisão (88%), todos os algoritmos têm um desempenho substancialmente bom.
Ao expandir o conjunto de dados adicionando todos os outros sensores relevantes, exceto a velocidade, para fins de economia de bateria, foram alcançados melhores resultados em termos de precisão. Com o segundo conjunto de dados, formado por oito sensores e trinta e dois recursos, a precisão aumenta até valores entre 86% e 93%.</p>
<p>Por fim, foi treinado um modelo no terceiro conjunto de dados formado por todos os nove sensores relevantes e trinta e seis recursos, diferindo do anterior apenas para recursos derivados de velocidade. O resultado mostra como se considera a velocidade, aumentando ainda mais a capacidade do modelo de inferir qual modo de transporte o usuário está usando no momento. Neste último caso, a precisão atingiu um nível de alcance entre 91% e 96%.</p>

<p>Acurácia geral com todos os quatro algoritmos de classificação do modelo de benchmark:</p>

| Algorithm | Accuracy on D1 | Accuracy on D2 | Accuracy of D3 |  
|---|:---:|:---:|:---:|
| Decision Tree (DT) | 76% | 78% | 86% |
| Random Forest (RF) | 81% | 89% | 93% |
| Support Vector Machine (SVM) | 76% | 86% | 90% |
| Neural Network (NN) | 76% | 87% | 91% | 


## III. Metodologia

### Pré-processamento de dados
<p>Optou-se pela utilização dos dados já balanceados, fornecidos pela equipe da Universidade de Bolonha. Nesses dados já houve um pré-processamento no sentido de equilibrar o número de amostras para cada um dos Modos de Transporte.</p>
<p>Da mesma forma, a equipe já fornece os dados com janelas de 5 segundos, equivalente a 1% dos dados brutos, e de 0,5 segundos, equivalente a 10% dos dados brutos.</p>
<p>Com esses dados, foram criadas 6 bases de dados, 3 para cada janela. O primeiro conjunto de dados teve a utilização das informações do acelerômetro, giroscópio e som. O segundo conjunto de dados teve as informações de 8 sensores e o terceiro conjunto de dados utilizou todos os nove sensores relevantes e trinta e seis recursos, diferindo do anterior apenas para recursos derivados de velocidade. Além dos atributos provenientes dos sensores, todas as bases de dados possuem o atributo tempo ('time').</p>
<p>Criadas as 6 bases de dados, foram tratados os valores faltantes com a ferramenta fillna() da biblioteca Pandas.</p>
<p>Após a divisão das bases de dados nas bases de treino e teste, foi utiliza a ferramenta MinMaxScaler da biblioteca Scikit-Learn, quando os dados foram escalados.</p>
<p>As variáveis dependentes ('y_train' e  'y_test') foram codificadas e, com a ferramenta to_categorical da biblioteca Keras, os dados de vetor de classe (integers) foram convertidos em matriz de classe binária.</p>
<p>Finalmente, as matrizes com as variáveis independentes ('X_train' e 'X_test') tiveram as suas formas expandidas com a inserção de um novo eixo com a ferramenta expand_dims da biblioteca Numpy.</p>

### Implementação
<li><strong>CNNs</strong></li>
<p>Inicialmente foi definida uma função evaluate_model(), que utiliza os dados de treino e teste, ajusta um modelo no conjunto de dados de treinamento, avalia-o no conjunto de dados de teste e retorna uma estimativa do desempenho do modelo.</p>
<p>Definiu-se o modelo CNN usando a biblioteca de aprendizagem profunda Keras, com duas camadas 1D CNN, seguidas por uma camada de Dropout para regularização e depois uma camada de Pooling.</p>
<p>A escolha pelas duas camadas 1D CNN se deu pela possibilidade de melhorar as chances de aprendizagem com os dados de entrada.</p>
<p>Como as CNNs aprendem rapidamente, a camada de Dropout desacelera um pouco o processo de aprendizado visando um melhor modelo final.</p>
<p>A camada de Pooling reduz os recursos aprendidos para 1/4 de seu tamanho, consolidando-os apenas nos elementos mais essenciais.</p>
<p>Após o CNN e o pooling, os recursos aprendidos são achatados em um vetor longo e passam por uma camada totalmente conectada antes da camada de saída usada para fazer uma previsão. A camada totalmente conectada idealmente fornece um buffer entre os recursos aprendidos e a saída com a intenção de interpretar os recursos aprendidos antes de fazer uma previsão.</p>
<p>Foram utilizadas configurações de 'filters' com tamanho 64 e um tamanho de kernel de 3. Os 'filters' são mapas de recursos, mais precisamente o número de vezes que a entrada é processada ou interpretada, enquanto o tamanho do kernel é o número de etapas de tempo de entrada em que a sequência de entrada é lida ou processada nos mapas de recursos.</p>
<p>A versão do gradiente descendente estocástico 'Adam' foi usada para otimizar a rede e também a função de perda de entropia cruzada categórica foi utilizada, uma vez que se trata de um problema de classificação de várias classes.</p>
<p>O modelo foi ajustado para um número fixo de épocas, neste caso 10, e um tamanho de lote de 32 amostras, onde 32 janelas de dados são expostas ao modelo antes que os pesos do modelo sejam atualizados.</p>
<p>Depois que o modelo é ajustado, ele é avaliado no conjunto de dados de teste e a precisão é retornada.</p>
<p>Entretanto, não podemos julgar a habilidade do modelo a partir de uma única avaliação.</p>
<p>A razão para isso é que as redes neurais são estocásticas, o que significa que a cada avaliação um modelo diferente resultará ao treinar a mesma configuração com os mesmos dados.</p>
<p>Esta é uma característica da rede na medida em que dá ao modelo sua capacidade adaptativa, mas requer uma avaliação um pouco mais complexa.</p>
<p>A avaliação do modelo é repetida um determinado número de vezes e, em seguida, é realizado um resumo do desempenho do modelo em cada uma dessas execuções. Por exemplo, evaluate_model () pode ser executado num total de 10 vezes. Isso resultará em uma população de pontuações de avaliação de modelo que devem ser resumidas.</p>
<p>A amostra de pontuações pode ser resumida calculando e relatando a média e o desvio padrão do desempenho. A média fornece a precisão média do modelo no conjunto de dados, enquanto o desvio padrão fornece a variância média da precisão da média.</p>

<li><strong>RNNs</strong></li>
<p>Inicialmente foi definida uma função evaluate_model(), que utiliza os dados de treino e teste, ajusta um modelo no conjunto de dados de treinamento, avalia-o no conjunto de dados de teste e retorna uma estimativa do desempenho do modelo.</p>
<p>Foi definido um modelo LSTM usando a biblioteca de aprendizagem profunda Keras, tendo uma única camada oculta.</p>
<p>Em seguida, uma camada de dropout destinada a reduzir o overfitting do modelo para os dados de treinamento.</p>
<p>Finalmente, uma camada densa totalmente conectada é usada para interpretar os recursos extraídos pela camada oculta do LSTM, antes que uma camada de saída final seja usada para fazer previsões.</p>
<p>O modelo é ajustado para um número fixo de épocas, neste caso 15, e um tamanho de lote de 64 amostras será usado, onde 64 janelas de dados serão expostas ao modelo antes que os pesos do modelo sejam atualizados.</p>
<p>Ajustado o modelo, ele é avaliado no conjunto de dados de teste e a precisão é retornada.</p>
<p>Entretanto, assim como nas CNNs, não podemos julgar a habilidade do modelo a partir de uma única avaliação.</p>
<p>A razão disso é que as redes neurais são estocásticas, o que significa que um modelo específico diferente resultará ao treinar a mesma configuração de modelo nos mesmos dados.</p>
<p>Esta é uma característica da rede na medida em que dá ao modelo sua capacidade adaptativa, mas requer uma avaliação um pouco mais complexa do modelo.</p>
<p>A avaliação do modelo é repetida várias vezes e, em seguida, é resumido o desempenho do modelo em cada uma dessas execuções. Por exemplo, pode ser chamado evaluate_model () um total de 10 vezes. Isso resultará em uma população de pontuações de avaliação de modelo que devem ser resumidas.</p>
<p>A amostra de pontuações pode ser resumida calculando e relatando a média e o desvio padrão do desempenho. A média fornece a precisão média do modelo no conjunto de dados, enquanto o desvio padrão fornece a variância média da precisão da média.</p>

### Refinamento
<li><strong>CNNs</strong></li>
<p>No processo de aperfeiçoamento do modelo, foram explorados vários hiperparâmetros..</p>
<p><strong>Número de filtros ('Filters')</strong></p>
<p>Um hiperparâmetro importante para a CNN é o número de mapas de filtro. Foram experimentados uma gama de valores diferentes, de menos a muitos mais do que os 64 usados no primeiro modelo desenvolvido.</p>
<p>Especificamente, foram testados os seguintes números de mapas de recursos:</p>
<p>n_params = [8, 16, 32, 64, 128, 256]</p>
<p><strong>Tamanho do Kernel</strong></p>
<p>O tamanho do kernel é outro hiperparâmetro importante da CND 1D para se ajustar.</p>
<p>O tamanho do kernel controla o número de etapas de tempo consideradas em cada “leitura” da sequência de entrada, que é então projetada no mapa de características (através do processo convolucional).</p>
<p>Um tamanho de kernel grande significa uma leitura menos rigorosa dos dados, mas pode resultar em um instantâneo mais generalizado da entrada.</p>
<p>Foi usada a mesma configuração experimental e testado um conjunto de diferentes tamanhos de kernel, além do padrão de três etapas de tempo. A lista completa de valores é a seguinte:</p>
<p>n_params = [2, 3, 5]</p>
<p><strong>Rede Neural Convolucional Multi-Headed</strong></p>
<p>Outra abordagem popular com CNNs é ter um modelo de múltiplas cabeças, onde cada cabeçote do modelo lê os passos de tempo de entrada usando um kernel de tamanho diferente.</p>
<p>Por exemplo, um modelo de três cabeças pode ter três tamanhos diferentes de kernel de 3, 5, 11, permitindo que o modelo leia e interprete os dados da sequência em três resoluções diferentes. As interpretações de todas as três cabeças são então concatenadas dentro do modelo e interpretadas por uma camada totalmente conectada antes que uma previsão seja feita.</p>
<p><strong>Número de épocas ('Epochs')</strong></p>
<p>O número de épocas é um hiperparâmetro que define o número de vezes que o algoritmo de aprendizado funcionará em todo o conjunto de dados de treinamento.</p>
<p>Uma época significa que cada amostra no conjunto de dados de treinamento teve a oportunidade de atualizar os parâmetros do modelo interno. Uma época é composta por um ou mais lotes.</p>
<p>Especificamente, foram testados os seguintes números épocas:</p>
<p>n_params = [10, 20, 40, 80]</p>
<p><strong>Tamanho do lote ('Batch Size')</strong></p>
<p>O tamanho do lote é um hiperparâmetro que define o número de amostras para trabalhar antes de atualizar os parâmetros internos do modelo.</p>
<p>Um lote pode ser pensado como um loop para iterar sobre uma ou mais amostras e fazer previsões. No final do lote, as previsões são comparadas com as variáveis de saída esperadas e um erro é calculado. A partir desse erro, o algoritmo de atualização é usado para melhorar o modelo, por exemplo, mover para baixo ao longo do gradiente de erro.</p>
<p>Foram testados os seguintes tamanhos de lote:</p>
<p>n_params = [32, 64, 128]</p>

<li><strong>RNNs</strong></li>
<p>Os modelos com RNN apresentaram um maior tempo de processamento e resultados inferiores em relação aos modelos com CNN.</p>
<p>Por isso, buscando um refinamento e no intuito de agregar as qualidades das CNNs, foi testada uma arquitetura CNN LSTM, que envolve o uso de camadas CNN (Convolutional Neural Network) para extração de características em dados de entrada combinados com LSTMs (RNN) para suportar a previsão de seqüências.

## IV. Resultados

### Modelo de avaliação e validação
Nesta seção, o modelo final e quaisquer qualidades que o sustentem devem ser avaliadas em detalhe. Deve ficar claro como o modelo final foi obtido e por que tal modelo foi escolhido. Além disso, algum tipo de análise deve ser realizada para validar a robustez do modelo e sua solução, como, por exemplo, manipular os dados de entrada ou o ambiente para ver como a solução do modelo é afetada (técnica chamada de análise sensitiva). Questões para se perguntar ao escrever esta seção:
- _O modelo final é razoável e alinhado com as expectativas de solução? Os parâmetros finais do modelo são apropriados?_
- _O modelo final foi testado com várias entradas para avaliar se o modelo generaliza bem com dados não vistos?_
-_O modelo é robusto o suficiente para o problema? Pequenas perturbações (mudanças) nos dados de treinamento ou no espaço de entrada afetam os resultados de forma considerável?_
- _Os resultados obtidos do modelo são confiáveis?_

### Justificativa

<li><strong>Benchmark</strong></li>
| Algorithm | Accuracy on D1 | Accuracy on D2 | Accuracy of D3 |  
|---|:---:|:---:|:---:|
| Decision Tree (DT) | 76% | 78% | 86% |
| Random Forest (RF) | 81% | 89% | 93% |
| Support Vector Machine (SVM) | 76% | 86% | 90% |
| Neural Network (NN) | 76% | 87% | 91% |

<li><strong>Projeto</strong></li>
| Algorithm | Accuracy on D1 | Accuracy on D2 | Accuracy of D3 | Accuracy of D3* |    
|---|:---:|:---:|:---:|:---:|
| Convolutional Neural Network (CNN) | 67% | 85% | 89% | 95% |
| Recurrent Neural Network (RNN) | 62% | 80% | 81% | 87% |
<p>* Janela de 0.5 segundos.</p>

Nesta seção, a solução final do seu modelo e os resultados dela obtidos devem ser comparados aos valores de referência (benchmark) que você estabeleceu anteriormente no projeto, usando algum tipo de análise estatística. Você deverá também justificar se esses resultados e a solução são significativas o suficiente para ter resolvido o problema apresentado no projeto. Questões para se perguntar ao escrever esta seção:
- _Os resultados finais encontrados são mais fortes do que a referência reportada anteriormente?_
- _Você analisou e discutiu totalmente a solução final?_
- _A solução final é significativa o suficiente para ter resolvido o problema?_


## V. Conclusão
_(aprox. 1-2 páginas)_

### Foma livre de visualização
Nesta seção, você deverá fornecer alguma forma de visualização que enfatize uma qualidade importante do projeto. A visualização é de forma livre, mas deve sustentar de forma razoável um resultado ou característica relevante sobre o problema que você quer discutir. Questões para se perguntar ao escrever esta seção:
- _Você visualizou uma qualidade importante ou relevante acerca do problema, conjunto de dados, dados de entrada, ou resultados?_
- _A visualização foi completamente analisada e discutida?_
- _Se um gráfico foi fornecido, os eixos, títulos e dados foram claramente definidos?_

### Reflexão
Nesta seção, você deverá resumir os procedimentos desde o problema até a solução e discutir um ou dois aspectos  do projeto que você achou particularmente interessante ou difícil. É esperado que você reflita sobre o projeto como um todo de forma a mostrar que você possui um entendimento sólido de todo o processo empregado em seu trabalho. Questões para se perguntar ao escrever esta seção:
- _Você resumiu inteiramente o processo que você utilizou neste projeto?_
- _Houve algum aspecto interessante do projeto?_
- _Houve algum aspecto difícil do projeto?_
- _O modelo e solução final alinham-se com suas expectativas para o problema, e devem ser usadas de forma geral para resolver esses tipos de problemas?_

### Melhorias
Nesta seção, você deverá discutir como um aspecto da sua implementação poderia ser melhorado. Por exemplo, considere maneiras de tornar a sua implementação mais geral e o que precisaria ser modificado. Você não precisa fazer a melhoria, mas as possíveis soluções que resultariam de tais mudanças devem ser consideradas e comparadas/contrastadas com a sua solução atual. Questões para se perguntar ao escrever esta seção:
- _Existem melhorias futuras que podem ser feitas nos algoritmos ou técnicas que você usou neste projeto?_
- _Existem algoritmos ou técnicas que você pesquisou, porém não soube como implementá-las, mas consideraria usar se você soubesse como?_
- _Se você usou sua solução final como nova referência, você acredita existir uma solução ainda melhor?_

-----------

**Antes de enviar, pergunte-se. . .**

- _O relatório de projeto que você escreveu segue uma estrutura bem organizada, similar ao modelo do projeto?_
- Cada seção (particularmente **Análise** e **Metodologia**) foi escrita de maneira clara, concisa e específica? Existe algum termo ou frase ambígua que precise de esclarecimento?
- O público-alvo do seu projeto será capaz de entender suas análises, métodos e resultados?
- Você revisou seu relatório de projeto adequadamente, de forma a minimizar a quantidade de erros gramaticais e ortográficos?
- Todos os recursos usados neste projeto foram corretamente citados e referenciados?
- O código que implementa sua solução está legível e comentado adequadamente?
- O código é executado sem erros e produz resultados similares àqueles reportados?
