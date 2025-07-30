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

 ## 🛠️ Como executar
   
Clone o repositório:
git clone https://github.com/Fabio4breu/lstm-previsao-acoes.git

cd lstm-previsao-acoes

Crie e ative um ambiente virtual:
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

Instale as dependências:
pip install -r requirements.txt

Execute:
python main.py

📝 Observações

Os dados são públicos e fornecidos via Yahoo Finance.

## 🧠 Exemplo de saída

```bash
🔍 Analisando AAPL...
📅 Último preço real:     US$ 185.31
🔮 Previsão próxima data: US$ 187.05
📉 Variação:              +1.74 (+0.94%)
📈 Tendência detectada:   Alta 📈
🧠 RMSE: 4.15 | MAE: 3.72
💡 Sugestão: POSSÍVEL BOA OPORTUNIDADE
