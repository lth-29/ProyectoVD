import streamlit as st
import time
import numpy as np
from funciones import *
from streamlit_folium import st_folium
from folium.plugins import TimestampedGeoJson
import folium


st.set_page_config(page_title="Animación", page_icon="🎞️", layout="wide")

variables = ["Gender Equality Index", "Work", "Money", "Knowledge", "Time", "Power", "Health"]

dominio = st.sidebar.selectbox("Dominio", variables)

data = load_data()

years, _ = get_years()

visualizacion = st.sidebar.radio("Tipo de visualizacion", ["Mapa", "Cartograma"])

animation(data, dominio, visualizacion)