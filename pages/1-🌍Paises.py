import streamlit as st
import time
import numpy as np
from funciones import *

st.set_page_config(page_title="Países", page_icon="🌍", layout="wide")

with open('style.css') as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

# Data
data = load_data()
variables = ["Gender Equality Index", "Work", "Money", "Knowledge", "Time", "Power", "Health"]

_, base = get_years()

# Sidebar
paises = sorted(list(set(data['Country'])))
paises.remove("Europe")
paises.insert(0, 'Europe')
country = st.sidebar.selectbox('País', paises, placeholder='Seleccione un país', index=None)

if country == None:
    st.title('Países')
    st.markdown('''
    En este apartado se mostrarán los datos de los países de la Unión Europea.
                
    Para visualizar los datos de un país en concreto, selecciónelo en el menú lateral.
    ''')
else:
    st.title(f'{country}')
    year = st.sidebar.selectbox('Año', sorted(list(set(data['Index Year']))), index=0)
    st.sidebar.markdown("")
    evolution = st.sidebar.checkbox('Evolución', value=False)

    # Data
    data_country = filter_data(data = data, year = year, countries = [country])

    years = sorted(list(set(data['Index Year'])))

    if data_country["Index Year"].values[0] != min(data["Index Year"]):
        last_year_index = years.index(data_country["Index Year"].values[0])-1
        st.markdown(f'''
        A continuación tienes información sobre los diferentes dominios de {country} en {year} y su diferencia respecto {years[last_year_index]}.
        ''')

    else:
        last_year_index = years.index(data_country["Index Year"].values[0])
        st.markdown(f'''
        A continuación tienes información sobre los diferentes dominios de {country} en {year}.
        ''')
    
    
    last_year = years[last_year_index]
    last_year = filter_data(data = data, year = last_year, countries = [country])

    st.markdown(f"""                
        Los datos del índice de {year} prodecen en su mayoría de {base[years.index(data_country["Index Year"].values[0])]}.
                """)

    # 8 grid columns
    cols = st.columns(7)
    for i, variable in enumerate(variables):
        col = cols[i % 7]
        # Calcular el valor
        var = round(data_country[variable].values[0].astype(float), 1)
        # Cálcular la diferencia respecto al año anterior
        last_var = round(var-last_year[variable].values[0].astype(float), 1)
        if last_var == 0:
            delta = ""
            color = "off"
            
        else:
            delta = last_var
            color = "normal"

        # card = make_cards(var, variable, delta)
        # col.markdown(card, unsafe_allow_html=True)
        col.metric(label=variable, value=var, delta=delta, delta_color=color)
    st.write("")

    if evolution:
        # data1 = data[(data['Country'] == country)].reset_index(drop=True)
        data1 = filter_data(data = data, countries = [country])
        show_graphs(data1, variables, "Lineas")
    if country != 'Europe':
        compare_europe = st.sidebar.checkbox('Comparar con Europa', value=False)
        if compare_europe:
            # data1 = data[(data['Country'] == country) & (data["Index Year"] == year)].reset_index(drop=True)
            data1 = filter_data(data = data, year = year, countries = [country])
            # data2 = data[(data['Country'] == 'Europe') & (data["Index Year"] == year)].reset_index(drop=True)
            data2 = filter_data(data = data, year = year, countries = ['Europe'])
            data2 = pd.concat([data2, data1], ignore_index=True)
            show_graphs(data2, variables, "BarraEurope")
    compare_domains = st.sidebar.checkbox('Comparar dominios', value=False)
    if compare_domains:
        # data1 = data[(data['Country'] == country)].reset_index(drop=True)
        data1 = filter_data(data = data, countries = [country])
        show_graphs(data1, variables, "Boxplot")

