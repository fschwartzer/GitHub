# Nanodegree Engenheiro de Machine Learning
## Proposta de projeto final
<p>Fernando Roberto Schwartzer</p>
<p>17 de novembro de 2018</p>

## Proposta

### Histórico do assunto
Identificar os modos de transporte através de observações dos usuários, ou observação do ambiente, é um tópico crescente de pesquisa, com muitas aplicações no planejamento da mobilidade urbana. A detecção do modo de transporte fornece informações para o diagnóstico do uso da malha viária, da ocupação do solo, do deslocamento de cargas e, principalmente, dos deslocamentos das pessoas nas cidades.
O reconhecimento do modo de transporte do usuário pode ser considerado como uma tarefa de HAR (Human Activity Recognition). Seu objetivo é identificar que tipo de transporte - caminhar, dirigir etc. - uma pessoa está usando.
Historicamente, os dados dos sensores para reconhecimento de atividades eram difíceis e caros de coletar, exigindo hardware personalizado. Agora, telefones inteligentes e outros dispositivos de rastreamento pessoal usados para monitoramento de saúde e fitness são baratos e onipresentes. Como tal, os dados de sensores destes dispositivos são mais baratos de coletar, mais comuns e, portanto, são uma versão mais comumente estudada do problema geral de reconhecimento de atividades.

### Descrição do problema
O problema consiste na previsão da atividade dada uma captura instantânea de dados, geralmente de um ou de um pequeno número de tipos de sensores. Geralmente, esse problema é enquadrado como uma tarefa de classificação de série temporal univariada ou multivariada.
É um problema desafiador, pois não há maneiras óbvias ou diretas de relacionar os dados do sensor registrado à atividades humanas específicas e cada sujeito pode realizar uma atividade com variação significativa, resultando em variações nos dados do sensor gravado.
A intenção é registrar os dados do sensor e as atividades correspondentes para assuntos específicos, ajustar um modelo a partir desses dados e generalizar o modelo para classificar a atividade de novos assuntos não vistos a partir de seus dados de sensor.

### Conjuntos de dados e entradas
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

### Descrição da solução
<p>A solução para a obtenção da melhor pontuação possível para a métrica Acurácia será a aplicação de modelos de redes neurais por convolução e modelos de redes neurais recorrentes.</p>
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

### Modelo de referência (benchmark)
<p>O modelo de referência será o trabalho de detecção de modo de transporte realizado por equipe da Universidade de Bolonha, Itália:</p>

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
<p>Por fim, foi treinado um modelo no terceiro conjunto de dados formado por todos os nove sensores relevantes e trinta e seis recursos, diferindo do anterior apenas para recursos derivados de velocidade. O resultado mostra como se considera a velocidade, aumentando ainda mais a capacidade do modelo de inferir qual modo de transporte o usuário está usando atualmente. Neste último caso, a precisão atingiu um nível de alcance entre 91% e 96%.</p>


### Métricas de avaliação
<p>A métrica de avaliação que a ser utilizada para quantificar o desempenho tanto do modelo de benchmark como do modelo de solução apresentados será a Acurácia.</p>

<img src= "https://github.com/fschwartzer/Udacity-Machine-Learning-Nanodegree/blob/master/Capstone/acuracia.png" />

<p>Acurácia geral com todos os quatro algoritmos de classificação do modelo de benchmark:</p>

| Algorithm | Accuracy on D1 | Accuracy on D2 | Accuracy of D3 |  
|---|:---:|:---:|:---:|
| Decision Tree (DT) | 76% | 78% | 86% |
| Random Forest (RF) | 81% | 89% | 93% |
| Support Vector Machine (SVM) | 76% | 86% | 90% |
| Neural Network (NN) | 76% | 87% | 91% | 


### Design do projeto
<p>No projeto serão utilizados 3 conjuntos de dados, aplicando 2 algoritmos.
Assim como no modelo de referência (benchmark), o primeiro conjunto de dados terá a utilização das informações do acelerômetro, giroscópio e som. O segundo conjunto de dados terá as informações de 8 sensores e o terceiro conjunto de dados de todos os nove sensores relevantes e trinta e seis recursos, diferindo do anterior apenas para recursos derivados de velocidade.
No projeto, serão aplicadas duas abordagens para as redes neurais, que são apropriadas para a classificação de séries temporais e que demonstraram ter um bom desempenho no reconhecimento de atividades usando dados de sensores de telefones inteligentes e dispositivos de rastreamento de condicionamento físico.
Serão os modelos de redes neurais por convolução (CNN) e modelos de redes neurais recorrentes (RNN).</p>
<p>O projeto estará estruturado da seguinte forma:</p>
<ul>
  <li>Exploração dos dados</li>
  <li>Pré-processamento dos dados</li>
  <li>Aplicação dos Modelos de CNN e RNN:</li>
  <li><Strong>CNN</Strong></li>
  <p><img src= "https://github.com/fschwartzer/Udacity-Machine-Learning-Nanodegree/blob/master/Capstone/Depiction-of-CNN-Model-for-Accelerompter-Data.png" | width=500 /></p>
  <li><Strong>RNN</Strong></li>
  <p><img src= "https://github.com/fschwartzer/Udacity-Machine-Learning-Nanodegree/blob/master/Capstone/Depiction-of-LSTM-RNN-for-Activity-Recognition.png" | width=500 /></p>
  <li>Escolha da melhor janela de dados</li>
  <li>Escolha do melhor modelo, visando uma Acurácia melhor que do modelo de referência (benchmark)</li>


-----------

**Referências**

- http://cs.unibo.it/projects/us-tm2017/index.html
- https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
- https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
- https://www.tensorflow.org/tutorials/estimators/cnn
- https://machinelearningmastery.com/deep-learning-models-for-human-activity-recognition/
