# 📊 Previsão de Ações com LSTM + Streamlit

Uma aplicação interativa para **previsão de preços de ações** usando **Redes Neurais LSTM** e **Streamlit**. Analise ações individualmente, em grupo ou em um ranking completo com recomendações.

---

## 🚀 Funcionalidades

- 🔮 Previsão do próximo preço com base nos últimos 60 dias.
- 📈📉 Tendência: Alta, Queda ou Estável.
- 🧠 Avaliação do modelo com RMSE e MAE.
- 📊 Gráficos interativos de preços e previsões.
- 🏆 Ranking de ações com melhores oportunidades e riscos.

---

## 🛠 Tecnologias

- Python 3  
- [NumPy](https://numpy.org/)  
- [Pandas](https://pandas.pydata.org/)  
- [yfinance](https://pypi.org/project/yfinance/)  
- [Matplotlib](https://matplotlib.org/)  
- [scikit-learn](https://scikit-learn.org/)  
- [TensorFlow](https://www.tensorflow.org/)  
- [Streamlit](https://streamlit.io/)  

---

## 💻 Instalação

1. **Clone o repositório:**

git clone https://github.com/Fabio4breu/lstm-previsao-acoes.git
cd lstm-previsao-acoes
Crie um ambiente virtual (opcional, mas recomendado):

python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
Instale as dependências:

pip install -r requirements.txt
▶️ Como executar

streamlit run main.py
A interface abrirá no navegador.

Selecione uma ação ou use Todas / Ranking.

Clique em Analisar para gerar previsões e gráficos.

📂 Estrutura do projeto

lstm-previsao-acoes/
│

├─ main.py             # Script principal com LSTM e Streamlit

├─ requirements.txt    # Dependências do projeto

├─ .venv/              # Ambiente virtual (não versionar)

└─ README.md           # Documentação

⚠️ Nota: Não é necessário versionar .venv, apenas requirements.txt.

⚙️ Observações
O modelo é treinado com 5 épocas para agilizar a execução. Aumente para resultados mais precisos.

GitHub: Fabio4breu

Projeto criado como estudo e prática de LSTM e Streamlit.
