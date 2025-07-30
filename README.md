# 📊 Previsão de Ações com LSTM

Este projeto utiliza redes neurais LSTM (Long Short-Term Memory) para prever preços de fechamento de ações com base em dados históricos coletados via Yahoo Finance.

## 🚀 Tecnologias Utilizadas

- Python
- Pandas, NumPy, Matplotlib
- Scikit-learn
- TensorFlow / Keras
- yfinance

## 📈 Como funciona?

1. Faz o download dos preços históricos das ações (AAPL, GOOGL, AMZN).
2. Normaliza os dados com `MinMaxScaler`.
3. Gera sequências de entrada com 60 dias anteriores.
4. Treina uma rede LSTM com duas camadas.
5. Avalia o modelo com RMSE e MAE.
6. Exibe a tendência e gera um gráfico de previsão vs. preço real.

## 🧠 Exemplo de saída

```bash
🔍 Analisando AAPL...
📅 Último preço real:     US$ 185.31
🔮 Previsão próxima data: US$ 187.05
📉 Variação:              +1.74 (+0.94%)
📈 Tendência detectada:   Alta 📈
🧠 RMSE: 4.15 | MAE: 3.72
💡 Sugestão: POSSÍVEL BOA OPORTUNIDADE
