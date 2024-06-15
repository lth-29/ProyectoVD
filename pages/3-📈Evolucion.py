import streamlit as st
import time
import numpy as np
from funciones import *
from streamlit_folium import st_folium

st.set_page_config(page_title="Evolución", page_icon="📈", layout="wide")
variables = ["Gender Equality Index", "Work", "Money", "Knowledge", "Time", "Power", "Health"]

with open('style.css') as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

data = load_data()

years, _ = get_years()
a_base = st.sidebar.selectbox("Año base", sorted(years))

a_comparar = st.sidebar.selectbox("Año a comparar", sorted(list(filter(lambda x: x > a_base, years))))

data = data[data["Index Year"].isin([a_base, a_comparar])]
data = data.melt(id_vars=["Country", "Index Year", "code"], var_name="Dominio", value_name="Valor")
data = data.pivot_table(index=["Country", "code", "Dominio"], columns=["Index Year"], values="Valor").reset_index()
data["Diferencia"] = round(data[a_comparar] - data[a_base], 3)
data["Index Year"] = f"{a_base} - {a_comparar}"



comparar = st.sidebar.radio("Comparar", ["Dominio concreto", "Todos los dominios"])

if comparar == "Dominio concreto":
    dominio = st.sidebar.selectbox("Dominio", variables)
    visualizaciones = ["Proporción", "Gráfico de barras", "Mapa", "Cartograma"]
    data = data[data["Dominio"] == dominio]
else:
    visualizaciones = ["Mapa", "Cartograma"]
    dominio = variables
    data = data[data["Dominio"].isin(dominio)]

visualizacion = st.sidebar.radio("Visualización", visualizaciones)

if (visualizacion == "Mapa" or visualizacion == "Cartograma") and comparar == ("Todos los dominios"):
    escala = st.sidebar.checkbox("Escala absoluta", value=False)
else:
    escala = False

st.title(f"Comparación entre {a_base} y {a_comparar}")

show_graphs(data, comparar, visualizacion, dominio, escala, pagina="Evolución")