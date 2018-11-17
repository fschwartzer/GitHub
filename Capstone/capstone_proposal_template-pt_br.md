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
  <li>- Marco Di Felice • Professor Associado • email: marco.difelice3@unibo.it</li>
  <li>- Luciano Bononi • Professor Associado • email: luciano.bononi@unibo.it</li>
  <li>- Luca Bedogni • Professor Assistente • email: luca.bedogni4@unibo.it</li>
  <li>- Vincenzo Lomonaco • Estudante de doutorado • email: vincenzo.lomonaco@unibo.it</li>
</ul>
<p>Colaboradores anteriores</p>
<ul>
  <li>- Claudia Carpineti • Mestranda • e-mail: claudia.carpineti@studio.unibo.it</li>
  <li>- Matteo Cappella • Aluno de mestrado • email: matteo.cappella@studio.unibo.it</li>
  <li>- Simone Passaretti • Aluno de mestrado • email: simone.passaretti@studio.unibo.it</li>
</ul>
<p>A coleta de dados foi controlada por um aplicativo Android em execução no telefone dos usuários enquanto eles realizavam atividades. Esse aplicativo, por meio de uma interface gráfica simples, permitiu que os voluntários gravassem seu nome, iniciassem e interrompessem a coleta de dados e rotulassem a atividade que estava sendo executada. Foi pedido aos usuários para usar o aplicativo durante atividades específicas, como caminhar, estar em um carro, em um trem, em um ônibus ou ficar parado. As atividades com estas abreviações:</p>
<p>T M = {bus, car, train, still, walking}</p> 
<p>O aplicativo registra cada evento do sensor com uma frequência máxima de 20 Hz. Os eventos ocorrem toda vez que um sensor detecta uma alteração nos parâmetros que está medindo, fornecendo quatro informações:</p>
<ul>
<li>- o nome do sensor que acionou o evento;</li>
<li>- o timestamp do evento;</li>
<li>- a acurácia do evento;</li>
<li>- os dados brutos do sensor que acionaram o evento.</li>
</ul>

#### Atributos
<ul>
<li>Id:</li>
<li>Time:</li>
<li>activityrecognition#0:</li>
<li>activityrecognition#1:</li>
<li>android.sensor.accelerometer#mean:</li>
<li>android.sensor.accelerometer#min:</li>
<li>android.sensor.accelerometer#max:</li>
<li>android.sensor.accelerometer#std:</li>
<li>android.sensor.game_rotation_vector#mean:</li>
<li>android.sensor.game_rotation_vector#min:</li>
<li>android.sensor.game_rotation_vector#max:</li>
<li>android.sensor.game_rotation_vector#std:</li>
<li>android.sensor.gravity#mean:</li>
<li>android.sensor.gravity#min:</li>
<li>android.sensor.gravity#max:</li>
<li>android.sensor.gravity#std:</li>
<li>android.sensor.gyroscope#mean:</li>
<li>android.sensor.gyroscope#min:</li>
<li>android.sensor.gyroscope#max:</li>
<li>android.sensor.gyroscope#std:</li>
<li>android.sensor.gyroscope_uncalibrated#mean:</li>
<li>android.sensor.gyroscope_uncalibrated#min:</li>
<li>android.sensor.gyroscope_uncalibrated#max:</li>
<li>android.sensor.gyroscope_uncalibrated#std:</li>
<li>android.sensor.light#mean:</li>
<li>android.sensor.light#min:</li>
<li>android.sensor.light#max:</li>
<li>android.sensor.light#std:</li>
<li>android.sensor.linear_acceleration#mean:</li>
<li>android.sensor.linear_acceleration#min:</li>
<li>android.sensor.linear_acceleration#max:</li>
<li>android.sensor.linear_acceleration#std:</li>
<li>android.sensor.magnetic_field#mean:</li>
<li>android.sensor.magnetic_field#min:</li>
<li>android.sensor.magnetic_field#max:</li>
<li>android.sensor.magnetic_field#std:</li>
<li>android.sensor.magnetic_field_uncalibrated#mean:</li>
<li>android.sensor.magnetic_field_uncalibrated#min:</li>
<li>android.sensor.magnetic_field_uncalibrated#max:</li>
<li>android.sensor.magnetic_field_uncalibrated#std:</li>
<li>android.sensor.orientation#mean:</li>
<li>android.sensor.orientation#min:</li>
<li>android.sensor.orientation#max:</li>
<li>android.sensor.orientation#std:</li>
<li>android.sensor.pressure#mean:</li>
<li>android.sensor.pressure#min:</li>
<li>android.sensor.pressure#max:</li>
<li>android.sensor.pressure#std:</li>
<li>android.sensor.proximity#mean:</li>
<li>android.sensor.proximity#min:</li>
<li>android.sensor.proximity#max:</li>
<li>android.sensor.proximity#std:</li>
<li>android.sensor.rotation_vector#mean:</li>
<li>android.sensor.rotation_vector#min:</li>
<li>android.sensor.rotation_vector#max:</li>
<li>android.sensor.rotation_vector#std:</li>
<li>android.sensor.step_counter#mean:</li>
<li>android.sensor.step_counter#min:</li>
<li>android.sensor.step_counter#max:</li>
<li>android.sensor.step_counter#std:</li>
<li>sound#mean:</li>
<li>sound#min:</li>
<li>sound#max:</li>
<li>sound#std:</li>
<li>speed#mean:</li>
<li>speed#min:</li>
<li>speed#max:</li>
<li>speed#std:</li>
<li>target:</li>
<li>user:</li>

### Descrição da solução
Modelos de redes neurais profundas estão alcançando resultados de ponta para o reconhecimento da atividade humana. Eles são capazes de realizar o aprendizado de recursos automáticos a partir dos dados brutos do sensor e os modelos de desempenho superior se ajustam a recursos específicos do domínio criados manualmente.

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
O modelo de referência será o trabalho de detecção de modo de transporte realizado por equipe da Universidade de Bolonha, Itália:
Carpineti C., Lomonaco V., Bedogni L., Di Felice M., Bononi L., "Custom Dual Transportation Mode Detection by Smartphone Devices Exploiting Sensor Diversity", in Proceedings of the 14th Workshop on Context and Activity Modeling and Recognition (IEEE COMOREA 2018), Athens, Greece, March 19-23, 2018
Pré-impressão disponível: https://arxiv.org/abs/1810.05596
Em seu trabalho, utilizaram 3 conjuntos de dados, aplicando 4 algoritmos.
Para cada conjunto, foram construídos quatro modelos com quatro algoritmos de classificação diferentes:
•	Decision Trees (DT)
•	Random Forest (RF)
•	Support Vector Machines(SVM)
•	Neural Network (NN)
Os sensores incluídos no primeiro conjunto (parâmetro 1) foram acelerômetro, som e giroscópio. Esses três sensores possuem os maiores valores de precisão obtidos individualmente.
O primeiro conjunto de dados é formado por doze recursos, quatro para cada sensor. Foi realizada a classificação com os quatro algoritmos de classificação mencionados anteriormente. A precisão geral dos algoritmos está entre 82% e 88%. Mesmo que a floresta aleatória produza os maiores valores de precisão (88%), todos os algoritmos têm um desempenho substancialmente bom.
Ao expandir o conjunto de dados adicionando todos os outros sensores relevantes, exceto a velocidade, para fins de economia de bateria, foram alcançados melhores resultados em termos de precisão. Com o segundo conjunto de dados, formado por oito sensores e trinta e dois recursos, a precisão aumenta até valores entre 86% e 93%.
Por fim, foi treinado um modelo no terceiro conjunto de dados formado por todos os nove sensores relevantes e trinta e seis recursos, diferindo do anterior apenas para recursos derivados de velocidade. O resultado mostra como se considera a velocidade, aumentando ainda mais a capacidade do modelo de inferir qual modo de transporte o usuário está usando atualmente. Neste último caso, a precisão atingiu um nível de alcance entre 91% e 96%.


### Métricas de avaliação
A métrica de avaliação que a ser utilizada para quantificar o desempenho tanto do modelo de benchmark como do modelo de solução apresentados será a Acurácia.
 

Acurácia geral com todos os quatro algoritmos de classificação do modelo de benchmark:

| Algorithm | Accuracy on D1 | Accuracy on D2 | Accuracy of D3 |  
|---|:---:|:---:|:---:|
| Decision Tree (DT) | 76% | 78% | 86% |
| Random Forest (RF) | 81% | 89% | 93% |
| Support Vector Machine (SVM) | 76% | 86% | 90% |
| Neural Network (NN) | 76% | 87% | 91% | 


### Design do projeto
No projeto serão utilizados 3 conjuntos de dados, aplicando 2 algoritmos.
Assim como no modelo de referência (benchmark), o primeiro conjunto de dados utilização as informações do acelerômetro, giroscópio e som. O segundo conjunto de dados terá as informações de 8 sensores e o terceiro conjunto de dados de todos os nove sensores relevantes e trinta e seis recursos, diferindo do anterior apenas para recursos derivados de velocidade.
No projeto, serão aplicadas duas abordagens para as redes neurais, que são apropriadas para a classificação de séries temporais e que demonstraram ter um bom desempenho no reconhecimento de atividades usando dados de sensores de telefones inteligentes e dispositivos de rastreamento de condicionamento físico.
Serão os modelos de redes neurais por convolução (CNN) e modelos de redes neurais recorrentes (RNN).

-----------

**Antes de enviar sua proposta, pergunte-se. . .**

- A proposta que você escreveu segue uma estrutura bem organizada, similar ao modelo de projeto?
- Todas as seções (em especial, **Descrição da solução** e **Design do projeto**) estão escritas de uma forma clara, concisa e específica? Existe algum termo ou frase ambígua que precise de esclarecimento?
- O público-alvo de seu projeto será capaz de entender sua proposta?
- Você revisou sua proposta de projeto adequadamente, de forma a minimizar a quantidade de erros gramaticais e ortográficos?
- Todos os recursos usados neste projeto foram corretamente citados e referenciados?
