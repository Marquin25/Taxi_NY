# Taxi_NY

# 🚕 Previsão de Tarifas de Táxi com Machine Learning

Este projeto tem como objetivo **prever o valor de tarifas de táxi** utilizando técnicas de Machine Learning.
O modelo final é um **ensemble (Voting Regressor)** combinando quatro algoritmos diferentes para melhorar a precisão das previsões.

---

## 📊 **Objetivo do Projeto**

Desenvolver um sistema capaz de estimar o preço esperado de uma corrida de táxi com base em variáveis como:

* Distância percorrida
* Horário da corrida
* Mês
* Número de passageiros

O foco foi obter um modelo **robusto, rápido e com baixo erro de previsão (MSE/RMSE)**.

---

## 🧹 **Preparação dos Dados**

O conjunto de dados passou por um processo cuidadoso de limpeza e engenharia de atributos:

* Conversão da data para *year, month, day, hour*
* Cálculo da distância usando a fórmula de **Haversine**
* Remoção de dados inconsistentes (corridas com distância zero, tarifas negativas etc.)
* Seleção das features:

  ```
  distance_km, hour, month, passenger_count
  ```

---

## 🤖 **Modelos Utilizados**

O ensemble utiliza quatro algoritmos distintos, cada um com características complementares:

### 🌲 **Random Forest (RF)**

Modelo baseado em diversas árvores de decisão.
É robusto e lida bem com dados complexos.

### 📈 **Linear Regression (LR)**

Modelo simples e interpretável.
Serve como base linear para comparar desempenho.

### 👥 **K-Nearest Neighbors (KNN)**

Usa os vizinhos mais próximos para prever valores.
Bom para padrões locais.

### 🌳 **Decision Tree (DT)**

Árvore de decisão única.
Ajuda na interpretação e captura relações não lineares.

---

## 🧠 **Ensemble: Voting Regressor**

Após treinar os modelos individualmente, foi aplicado um **VotingRegressor** combinando:

* RF
* LR
* KNN
* DT

O ensemble retorna a **média ponderada das previsões**, entregando resultados mais estáveis e precisos.

```python
ensemble = VotingRegressor([
    ('RF', model_rf),
    ('LR', model_lr),
    ('KNN', model_knn),
    ('DT', model_dt)
])
```

---

## 📈 **Métricas Avaliadas**

### 🟦 **MSE – Mean Squared Error**

Mede o erro quadrático médio. Quanto menor, melhor.

### 🟥 **RMSE – Root Mean Squared Error**

Raiz do MSE; mostra o erro na mesma escala da variável alvo (tarifa).

---

## 🔧 **Otimização de Hiperparâmetros**

Para melhorar o desempenho do Random Forest foi feita uma busca usando:

* **Bayesian Optimization (skopt)**
* Função objetivo com K-Fold Cross Validation

Isso garantiu um modelo mais bem ajustado e com menor overfitting.

---

## 🚀 **Como Executar o Projeto**

1. Instalar dependências

   ```
   pip install scikit-learn pandas numpy matplotlib scikit-optimize
   ```
2. Carregar o dataset
3. Rodar o notebook ou script principal
4. Ajustar os parâmetros conforme necessário

---

## 📌 **Resultados**

O ensemble apresentou **melhor desempenho** que os modelos individuais, obtendo um MSE menor e previsões mais estáveis.

Link Google Colab: https://colab.research.google.com/drive/1V3WJ9pNAd18e3LVaTCMvwNQM85dbHhdO?usp=sharing


