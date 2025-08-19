# ============================================
# 📊 Previsão de Ações com LSTM + Streamlit
# ============================================

# Importando bibliotecas essenciais
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import streamlit as st

# -----------------------------
# Configurações iniciais
# -----------------------------
tickers = ['AAPL', 'GOOGL', 'AMZN']  # Lista de ações a serem analisadas
seq_length = 60                       # Quantidade de dias usados para prever o próximo
start_date = '2015-01-01'             # Data inicial para download dos preços
end_date = '2024-12-31'               # Data final


# -----------------------------
# Funções auxiliares
# -----------------------------
def create_sequences(data, seq_length):
    """
    Cria sequências de dados para alimentar a rede LSTM.
    Exemplo: últimos 60 dias -> prevê o próximo valor.
    """
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i, 0])  # Sequência
        y.append(data[i, 0])                # Valor alvo
    return np.array(X), np.array(y)


def analisar_acao(ticker):
    """
    Faz o download da ação escolhida, treina a LSTM,
    gera previsões, calcula métricas e retorna resultados.
    """

    # ---- 1. Download dos dados históricos ----
    df = yf.download(ticker, start=start_date, end=end_date)[['Close']].dropna()

    # Normaliza os dados para valores entre 0 e 1 (necessário p/ LSTM)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Cria sequências de entrada e saída
    X, y = create_sequences(scaled_data, seq_length)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # formato exigido pela LSTM

    # Divisão treino/teste (80% treino, 20% teste)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # ---- 2. Construção do modelo LSTM ----
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))  # saída: previsão do próximo valor
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Treinamento (poucas épocas para agilizar)
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    # ---- 3. Previsões ----
    y_pred = model.predict(X_test)
    y_pred_scaled = scaler.inverse_transform(y_pred)  # desfaz normalização
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # ---- 4. Avaliação do modelo ----
    rmse = np.sqrt(mean_squared_error(y_test_scaled, y_pred_scaled))
    mae = mean_absolute_error(y_test_scaled, y_pred_scaled)

    # Último preço real e próxima previsão
    ultimo_real = float(y_test_scaled[-1])
    proxima_previsao = float(y_pred_scaled[-1])

    # Diferença e percentual
    variacao = proxima_previsao - ultimo_real
    variacao_pct = (variacao / ultimo_real) * 100

    # Tendência e sugestão
    if variacao > 1:
        tendencia = 'Alta 📈'
        sugestao = 'POSSÍVEL BOA OPORTUNIDADE'
    elif variacao < -1:
        tendencia = 'Queda 📉'
        sugestao = 'RISCO DE QUEDA'
    else:
        tendencia = 'Estável ⚖️'
        sugestao = 'NEUTRO - OBSERVAR'

    # ---- 5. Gráfico individual ----
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(y_test_scaled, label='Preço Real')
    ax.plot(y_pred_scaled, label='Previsão LSTM')
    ax.set_title(f'{ticker} - Preço x Previsão')
    ax.set_xlabel('Tempo')
    ax.set_ylabel('Preço')
    ax.legend()
    ax.grid(True)

    # Retorna tudo em formato de dicionário
    return {
        "ticker": ticker,
        "ultimo_real": ultimo_real,
        "proxima_previsao": proxima_previsao,
        "variacao": variacao,
        "variacao_pct": variacao_pct,
        "tendencia": tendencia,
        "sugestao": sugestao,
        "rmse": rmse,
        "mae": mae,
        "grafico": fig
    }


# -----------------------------
# Interface Streamlit
# -----------------------------
st.title("📊 Previsão de Ações com LSTM")

# Menu de opções
opcao = st.selectbox("Escolha uma ação:", tickers + ["Todas", "Ranking"])

# Botão de execução
if st.button("Analisar"):

    # Caso escolha "Todas" → roda em cada ação
    if opcao == "Todas":
        for t in tickers:
            resultado = analisar_acao(t)
            st.subheader(f"📌 {resultado['ticker']}")
            st.write(f"📅 Último preço real: **US$ {resultado['ultimo_real']:.2f}**")
            st.write(f"🔮 Previsão próxima data: **US$ {resultado['proxima_previsao']:.2f}**")
            st.write(f"📉 Variação: {resultado['variacao']:+.2f} ({resultado['variacao_pct']:+.2f}%)")
            st.write(f"📈 Tendência: {resultado['tendencia']}")
            st.write(f"🧠 RMSE: {resultado['rmse']:.2f} | MAE: {resultado['mae']:.2f}")
            st.success(f"💡 {resultado['sugestao']}")
            st.pyplot(resultado["grafico"])

    # Caso escolha "Ranking" → gera tabela e comparações
    elif opcao == "Ranking":
        resultados = []
        for t in tickers:
            resultados.append(analisar_acao(t))

        # Monta DataFrame com resumo
        df_resultados = pd.DataFrame([{
            "Ticker": r["ticker"],
            "Último Preço": r["ultimo_real"],
            "Previsão": r["proxima_previsao"],
            "Variação (%)": r["variacao_pct"],
            "Tendência": r["tendencia"],
            "Sugestão": r["sugestao"]
        } for r in resultados])

        # Mostra tabela geral
        st.subheader("📊 Ranking de Ações")
        st.dataframe(df_resultados.style.format({"Último Preço": "{:.2f}", "Previsão": "{:.2f}", "Variação (%)": "{:.2f}"}))

        # Separa por tendência
        st.markdown("### 📈 Melhores oportunidades")
        melhores = df_resultados[df_resultados["Tendência"].str.contains("Alta")]
        if not melhores.empty:
            st.table(melhores[["Ticker", "Previsão", "Variação (%)", "Sugestão"]])
        else:
            st.warning("Nenhuma ação em tendência de alta no momento.")

        st.markdown("### ⚠️ Risco de queda")
        piores = df_resultados[df_resultados["Tendência"].str.contains("Queda")]
        if not piores.empty:
            st.table(piores[["Ticker", "Previsão", "Variação (%)", "Sugestão"]])
        else:
            st.info("Nenhuma ação em queda significativa no momento.")

        # ---- NOVO: Gráfico comparativo de variações ----
        st.markdown("### 📊 Comparação das variações (%) entre ações")
        fig_bar, ax_bar = plt.subplots(figsize=(7, 4))
        ax_bar.bar(df_resultados["Ticker"], df_resultados["Variação (%)"], color="skyblue")
        ax_bar.axhline(0, color="red", linestyle="--")  # linha de referência
        ax_bar.set_title("Variação percentual prevista")
        ax_bar.set_ylabel("Variação (%)")
        st.pyplot(fig_bar)

    # Caso escolha apenas 1 ação específica
    else:
        resultado = analisar_acao(opcao)
        st.subheader(f"📌 {resultado['ticker']}")
        st.write(f"📅 Último preço real: **US$ {resultado['ultimo_real']:.2f}**")
        st.write(f"🔮 Previsão próxima data: **US$ {resultado['proxima_previsao']:.2f}**")
        st.write(f"📉 Variação: {resultado['variacao']:+.2f} ({resultado['variacao_pct']:+.2f}%)")
        st.write(f"📈 Tendência: {resultado['tendencia']}")
        st.write(f"🧠 RMSE: {resultado['rmse']:.2f} | MAE: {resultado['mae']:.2f}")
        st.success(f"💡 {resultado['sugestao']}")
        st.pyplot(resultado["grafico"])
