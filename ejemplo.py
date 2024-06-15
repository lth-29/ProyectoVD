import streamlit as st
import pandas as pd
import plotly.express as px
from funciones import load_data

# Cargar los datos del Gender Equality Index
# Suponemos que tienes un archivo CSV con estos datos
# @st.cache
# def load_data():
#     data = pd.read_csv('Gender Equality Index.csv')
#     return data

data = load_data()

# Título del dashboard
st.title("Dashboard de Índice de Igualdad de Género")

# Sección de KPI
st.header("Indicadores Clave de Desempeño")
average_index = data['Gender Equality Index'].mean()
max_index_Country = data.loc[data['Gender Equality Index'].idxmax()]['Country']
min_index_Country = data.loc[data['Gender Equality Index'].idxmin()]['Country']

st.metric("Índice Promedio de Igualdad de Género en Europa", f"{average_index:.2f}")
st.metric("País con el Índice más Alto", max_index_Country)
st.metric("País con el Índice más Bajo", min_index_Country)

# Gráfico de Líneas Temporal
st.header("Evolución Temporal del Índice de Igualdad de Género")
fig_line = px.line(data, x='Index Year', y='Gender Equality Index', color='Country', title='Evolución del Índice de Igualdad de Género')
st.plotly_chart(fig_line)

# Mapa de Calor
st.header("Mapa de Calor del Índice de Igualdad de Género")
fig_map = px.choropleth(data, locations="Country", color="Gender Equality Index", 
                        hover_name="Country", animation_frame="Index Year", title="Mapa de Calor del Índice de Igualdad de Género")
st.plotly_chart(fig_map)

# Gráfico de Barras Comparativo
st.header("Comparación del Índice de Igualdad de Género por País")
latest_Index = data['Index Year'].max()
latest_data = data[data['Index Year'] == latest_Index]
fig_bar = px.bar(latest_data, x='Country', y='Gender Equality Index', title=f'Índice de Igualdad de Género en {latest_Index}')
st.plotly_chart(fig_bar)

# Gráfico de Radar
st.header("Comparación de Dimensiones del Índice de Igualdad de Género")
dimensions = ["Work", "Money", "Knowledge", "Time", "Power", "Health"]
# fig_radar = px.line_polar(latest_data, r=dimensions, theta=dimensions, line_close=True, color='Country', title='Comparación de Dimensiones')
# st.plotly_chart(fig_radar)

# Gráfico de Dispersión
st.header("Relación entre Trabajo y Poder en el Índice de Igualdad de Género")
fig_scatter = px.scatter(latest_data, x='Work', y='Power', color='Country', title='Relación entre Trabajo y Poder')
st.plotly_chart(fig_scatter)

# Diagrama de Caja
st.header("Distribución del Índice de Igualdad de Género por País")
fig_box = px.box(latest_data, y='Gender Equality Index', points='all', title='Distribución del Índice de Igualdad de Género')
st.plotly_chart(fig_box)

# Panel de Detalle Interactivo
st.header("Detalles por País")
Country = st.selectbox('Selecciona un país', data['Country'].unique())
Country_data = data[data['Country'] == Country]
fig_detail = px.line(Country_data, x='Index Year', y=dimensions, title=f'Detalle del Índice de Igualdad de Género para {Country}')
st.plotly_chart(fig_detail)
