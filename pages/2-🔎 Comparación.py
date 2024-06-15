import streamlit as st
import time
import numpy as np
from funciones import *
from streamlit_folium import st_folium

st.set_page_config(page_title="Comparación", page_icon="🔎", layout="wide")

with open('style.css') as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

data = load_data()

variables = variables = ["Gender Equality Index", "Work", "Money", "Knowledge", "Time", "Power", "Health"]


dominio = st.sidebar.selectbox("Dominio a comparar", variables)

tipo = st.sidebar.radio("Tipo de comparación", ["Año concreto", "Múltiples años"])

if tipo == "Año concreto":
    ops = ["Gráfico de barras", "Mapa", "Cartograma"]
else:
    ops = ["Heatmap", "Mapa", "Cartograma"]

visualizacion = st.sidebar.radio("Visualización", ops)

if (visualizacion == "Mapa" or visualizacion == "Cartograma") and tipo == "Múltiples años":
    escala = st.sidebar.checkbox("Escala absoluta", value=False)
else:
    escala = False

if tipo == "Año concreto":
    year = st.sidebar.selectbox('Año', sorted(list(set(data['Index Year']))), index=0)
    # data = data[data["Index Year"] == year]
    data = filter_data(data, variables = [dominio], year = year)

elif tipo == "Múltiples años" and visualizacion == "Mapa":
    onecountry = st.sidebar.checkbox("País concreto", value=False)

    if onecountry:

        paises = sorted(list(set(data['Country'])))
        paises.remove("Europe")
        pais = st.sidebar.selectbox("País", paises, index=0)

        data2 = data.copy()
        data = filter_data(data,countries=[pais, "Europe"])

# si no existe data2
try:
    type(data2)
except NameError:
    data2 = None

c = st.columns(1)

with c[0]:
    custom_visualization(data, visualizacion, tipo, dominio, escala, auxiliar=data2, pagina="Comparación")


