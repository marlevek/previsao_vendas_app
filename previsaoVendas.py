import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title='Previsão de Vendas', page_icon=':chart:')

# Função para carregar e processar os dados
def load_data(file):
    df = pd.read_csv(file)
    df['datum'] = pd.to_datetime(df['datum'], format='%Y-%m-%d')
    return df

# Sidebar
with st.sidebar:
    upload_file = st.file_uploader('Escolha o arquivo', type=['csv'])
    
    # Checar se o arquivo foi carregado
    if upload_file is not None:
        data = load_data(upload_file)
        columns = [col for col in data.columns if col != 'datum']
    else:
        data = pd.DataFrame()
        columns = []

# Mostra as 10 primeiras linhas
st.subheader("Dados de Vendas - Primeiras 10 linhas")
if not data.empty:
    st.write(data.head(10))

# Sidebar para seleção do fármaco
selected_drug = st.sidebar.selectbox('Selecione o fármaco', columns)

# Função para prever vendas com ARIMA
def forecast_arima(df, column, periods):
    df = df.set_index('datum')
    df = df[column].asfreq('D').fillna(0)  # Preencher valores ausentes e definir frequência diária
    model = ARIMA(df, order=(5, 1, 0))  # Parâmetros do modelo ARIMA podem ser ajustados
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    return forecast

st.title('Previsão de Vendas de Fármacos')

if selected_drug:
    st.write(f'Previsão de vendas para: {selected_drug}')
    
    # Número de períodos para prever
    periods = st.slider("Número de períodos (dias) para prever", min_value=1, max_value=365, value=30)
    
    # Forecast
    try:
        forecast = forecast_arima(data, selected_drug, periods)
        st.write(f'Previsão para os próximos {periods} dias:')
        forecast_index = pd.date_range(start=data['datum'].max() + pd.Timedelta(days=1), periods=periods)
        forecast_series = pd.Series(forecast, index=forecast_index)
        st.line_chart(forecast_series)
    except Exception as e:
        st.error(f"Erro ao realizar a previsão: {e}")
