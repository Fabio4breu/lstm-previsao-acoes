import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -----------------------------
# Configurações iniciais
# -----------------------------
tickers = ['AAPL', 'GOOGL', 'AMZN']  # Lista de ações para analisar
seq_length = 60
start_date = '2015-01-01'
end_date = '2024-12-31'

# -----------------------------
# Funções auxiliares
# -----------------------------
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def analisar_acao(ticker):
    print(f"\n🔍 Analisando {ticker}...")

    df = yf.download(ticker, start=start_date, end=end_date)[['Close']].dropna()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    X, y = create_sequences(scaled_data, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Modelo LSTM
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Previsão
    y_pred = model.predict(X_test)
    y_pred_scaled = scaler.inverse_transform(y_pred)
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Avaliação
    rmse = np.sqrt(mean_squared_error(y_test_scaled, y_pred_scaled))
    mae = mean_absolute_error(y_test_scaled, y_pred_scaled)

    ultimo_real = float(y_test_scaled[-1])
    proxima_previsao = float(y_pred_scaled[-1])
    variacao = proxima_previsao - ultimo_real
    variacao_pct = (variacao / ultimo_real) * 100

    if variacao > 1:
        tendencia = 'Alta 📈'
        sugestao = 'POSSÍVEL BOA OPORTUNIDADE'
    elif variacao < -1:
        tendencia = 'Queda 📉'
        sugestao = 'RISCO DE QUEDA'
    else:
        tendencia = 'Estável ⚖️'
        sugestao = 'NEUTRO - OBSERVAR'

    # Exibir resultado no terminal
    print(f"📅 Último preço real:     US$ {ultimo_real:.2f}")
    print(f"🔮 Previsão próxima data: US$ {proxima_previsao:.2f}")
    print(f"📉 Variação:              {variacao:+.2f} ({variacao_pct:+.2f}%)")
    print(f"📈 Tendência detectada:   {tendencia}")
    print(f"🧠 RMSE: {rmse:.2f} | MAE: {mae:.2f}")
    print(f"💡 Sugestão: {sugestao}")

    # Gráfico
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_scaled, label='Preço Real')
    plt.plot(y_pred_scaled, label='Previsão LSTM')
    plt.title(f'{ticker} - Preço x Previsão')
    plt.xlabel('Tempo')
    plt.ylabel('Preço')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Executar para todas as ações
# -----------------------------
for ticker in tickers:
    analisar_acao(ticker)
