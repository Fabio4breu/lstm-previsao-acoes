# 📊 Previsão de Ações com LSTM + Streamlit

Este projeto utiliza **Redes Neurais LSTM** para previsão de preços de ações e fornece uma interface interativa com **Streamlit**. É possível analisar ações individualmente, todas as ações de uma vez ou gerar um ranking com recomendações.

---

## 🔹 Funcionalidades

- Previsão do próximo preço com base nos últimos 60 dias.
- Avaliação do modelo com métricas RMSE e MAE.
- Indicação de tendência: Alta 📈, Queda 📉 ou Estável ⚖️.
- Sugestões baseadas na previsão (Oportunidade, Risco, Neutro).
- Gráficos interativos de preços e previsões.
- Ranking de ações comparando variações percentuais.

---

## 🔹 Tecnologias e Bibliotecas

- Python 3
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [yfinance](https://pypi.org/project/yfinance/)
- [Matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Streamlit](https://streamlit.io/)

---

## 🔹 Instalação

1. **Clone o repositório:**

git clone https://github.com/Fabio4breu/lstm-previsao-acoes.git
cd lstm-previsao-acoes
Crie um ambiente virtual (opcional, mas recomendado):

bash
Copiar
Editar
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
Instale as dependências:

bash
Copiar
Editar
pip install -r requirements.txt
🔹 Como executar
bash
Copiar
Editar
streamlit run main.py
A interface abrirá no navegador.

Escolha uma ação ou use as opções “Todas” ou “Ranking” para analisar múltiplas ações.

Clique em Analisar para gerar previsões e gráficos.

🔹 Estrutura do Projeto
bash
Copiar
Editar
lstm-previsao-acoes/
│
├─ main.py             # Script principal com LSTM e interface Streamlit
├─ requirements.txt    # Dependências do projeto
├─ .venv/              # Ambiente virtual (não é necessário versionar)
└─ README.md           # Documentação
Nota: Recomenda-se não versionar o diretório .venv, apenas requirements.txt.

🔹 Observações
O modelo LSTM é treinado com poucas épocas (5) para agilizar a execução. Para resultados mais precisos, aumente o número de épocas no model.fit().

As previsões são baseadas em dados históricos e não garantem lucro ou precisão absoluta. Use apenas para estudo ou referência.

🔹 Autor
Fábio Antônio Santos Abreu

GitHub: Fabio4breu

Projeto criado como estudo e aplicação prática de LSTM e Streamlit.

yaml
Copiar
Editar

---

Se você quiser, posso **adaptar esse README** para que fique **mais visual**, com emojis, seções dobráveis e instruções resumidas para quem for abrir no GitHub, deixando ele bem profissional.  

Quer que eu faça essa versão aprimorada?







Você disse:
sim


O ChatGPT disse:
Perfeito! Aqui está uma versão do README.md mais visual, organizada, com emojis e instruções claras para facilitar a leitura no GitHub:

markdown
Copiar
Editar
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

bash
Copiar
Editar
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
Instale as dependências:

bash
Copiar
Editar
pip install -r requirements.txt
▶️ Como executar
bash
Copiar
Editar
streamlit run main.py
A interface abrirá no navegador.

Selecione uma ação ou use Todas / Ranking.

Clique em Analisar para gerar previsões e gráficos.

📂 Estrutura do projeto
bash
Copiar
Editar
lstm-previsao-acoes/
│
├─ main.py             # Script principal com LSTM e Streamlit
├─ requirements.txt    # Dependências do projeto
├─ .venv/              # Ambiente virtual (não versionar)
└─ README.md           # Documentação
⚠️ Nota: Não é necessário versionar .venv, apenas requirements.txt.

⚙️ Observações
O modelo é treinado com 5 épocas para agilizar a execução. Aumente para resultados mais precisos.

As previsões são educacionais e não garantem retorno financeiro.
