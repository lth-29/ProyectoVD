import folium.plugins
import pandas as pd
import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from shapely.geometry import Polygon
import pycountry
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_hex


default_color="#a2d2ff"

def get_text_color(rgba_color):
    """
    Determina el color del texto (negro o blanco) basado en la luminancia del color de fondo.

    Parameters:
    rgba_color (tuple): A tuple representing the RGBA color values.

    Returns:
    str: The text color, either "black" or "white".
    """
    r, g, b = rgba_color[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance > 0.5 else "white"

def get_color(value, cmap, norm):
    '''
    Obtener el color en hexadecimal y los valores RGB de un valor en función de un mapa de colores y una normalización.

    Argumentos:
        value (float): El valor a mapear.
        cmap (matplotlib.colors.Colormap): El mapa de colores a utilizar.
        norm (matplotlib.colors.Normalize): La normalización a utilizar.

    Devuelve:
        tuple: Una tupla con el color en hexadecimal y los valores RGB.

    '''
    rgba = cmap(norm(value))
    color_hex = to_hex(rgba)
    return color_hex, rgba[:3]

def get_years():
    years = [2013, 2015, 2017, 2019, 2020, 2021, 2022, 2023]
    base = [2010, 2012, 2015, 2017, 2018, 2019, 2020, 2021]
    return years, base


@st.cache_data
def load_cartography():
    # geojson 
    shape = gpd.read_file('./data/NUTS_RG_20M_2021_4326.shp.zip')
    polygon_coords = [(-20, -35), (-20, 85), (85, 50), (-35, -20)]
    polygon = Polygon(polygon_coords)
    # Filtrar el GeoDataFrame utilizando el polígono como máscara
    df = shape.explode().explode()
    df = df[df.geometry.intersects(polygon)]
    df = df.dissolve(by='CNTR_CODE')
    df = df.reset_index()
    eng = []
    for i in range(len(df)):
        if df.loc[i, 'CNTR_CODE'] == 'EL':
            df.loc[i, 'CNTR_CODE'] = 'GR'
        elif df.loc[i, 'CNTR_CODE'] == 'UK':
            df.loc[i, 'CNTR_CODE'] = 'GB'
        eng.append(pycountry.countries.get(alpha_2=df.loc[i, 'CNTR_CODE']).name)
    df['Eng'] = eng
    return df

def load_countries():
    with open('data/euro-countries.json') as f:
        countries = json.load(f)
    # Create a DataFrame
    countries = pd.DataFrame.from_records(countries)
    countries.rename(columns={'name': 'Country'}, inplace=True)
    return countries

def load_coords():
    coords = pd.read_csv('data/EU27-coord.csv', sep=';', decimal=',')
    coords.loc[coords['CNTR_CODE'] == 'EL', 'CNTR_CODE'] = 'GR'
    coords = pd.concat([coords, pd.DataFrame([["EU", 2, 12]], columns=['CNTR_CODE', 'lng', 'lat'])], ignore_index=True)
    return coords

def load_data():
    # Load the data

    years, base = get_years()
    df = pd.DataFrame()

    for year, base in zip(years, base):
        page = f'{year} ({base} data)'
        data = pd.read_excel("data/gender-equality-index.xlsx", sheet_name=page, skiprows=1)
        df = pd.concat([df, data], ignore_index=True)
    
    # Columnas en formato title
    df.columns = df.columns.str.title()

    # Load the countries
    countries = load_countries()

    # Arreglar nombres de Grecia
    df.loc[df['Country'] == 'EL', 'Country'] = 'GR'

    df = pd.merge(df, countries, how='left', left_on='Country', right_on='code')
    df.drop(columns=['Country_x'], inplace=True)
    df = df.rename(columns={'Country_y': 'Country'})

    # Renombrar EU27 a EU
    df['Country'] = df['Country'].replace(np.nan, 'Europe')
    df['code'] = df['code'].replace(np.nan, 'EU')

    # Return the data
    return df

def filter_data(data, year=None, countries=None, variables=None):
    if year:
        data = data[data['Index Year'] == year]
    if countries:
        data = data[data['Country'].isin(countries)]
    if variables:
        v = ["Country", "Index Year", "code"] + variables
        data = data.loc[:, v]
    return data.reset_index(drop=True)


def square_vertices(cx, cy):
    return [
        (cx - 0.5, cy - 0.5), # abajo izq
        (cx + 0.5, cy - 0.5), # abajo der
        (cx + 0.5, cy + 0.5), # arr der
        (cx - 0.5, cy + 0.5), #ar izq
    ]

def custom_metrics(data):
    data = data[data['Country'] != 'Europe']
    pos = data[data['Diferencia'] > 0].shape[0]
    neg = data[data['Diferencia'] < 0].shape[0]

    if data[data['Diferencia'] > 0].shape[0] == 0:
        media_pos, min_pos, max_pos = 0, 0, 0

    else:
        media_pos = data[data['Diferencia'] > 0]['Diferencia'].mean().round(3)
        max_pos = data[data['Diferencia'] > 0]['Diferencia'].max().round(3)
        min_pos = data[data['Diferencia'] > 0]['Diferencia'].min().round(3)
    
    if data[data['Diferencia'] < 0].shape[0] == 0:
        media_neg, min_neg, max_neg = 0, 0, 0

    else:
        media_neg = data[data['Diferencia'] < 0]['Diferencia'].mean().round(3)
        max_neg = data[data['Diferencia'] < 0]['Diferencia'].min().round(3)
        min_neg = data[data['Diferencia'] < 0]['Diferencia'].max().round(3)

    cols = st.columns(2)
    with cols[0]:
        # st.metric(label="Países con mejora", value=pos)
        st.metric(label="Media de mejora", value=media_pos)
        st.metric(label="Máxima mejora", value=max_pos)
        st.metric(label="Mínima mejora", value=min_pos)
    with cols[1]:
        # st.metric(label="Países con empeoramiento", value=neg)
        st.metric(label="Media de empeoramiento", value=media_neg)
        st.metric(label="Máximo empeoramiento", value=max_neg)
        st.metric(label="Mínimo empeoramiento", value=min_neg)

def show_graphs(data, variables, type="", d = None, escala=False, pagina = None):
    # Create the graphs
    fig = go.Figure()

    if type == "Lineas":

        df = data.melt(id_vars=['Country', 'code', 'Index Year'], var_name='Variable', value_name='Value')
        df = df[df['Variable'].isin(variables)]
        # df = df.sort_values(by='Variable', ascending=True)
        
        fig = px.line(df, x='Index Year', y='Value', color='Variable', 
                    hover_data=['Variable'], markers=True, line_shape='linear',
                    labels={'Index Year': 'Año', 'Value': 'Valor', "Variable": "Dominio"},
                    category_orders={'Variable': sorted(variables)},
                    )
        fig.update_layout(
            title=f'Evolución de los dominios en {data["Country"].values[0]}',
            xaxis_title='Año',
            yaxis_title='Valor del dominio',
            xaxis=dict(tickmode='linear'),
            grid=dict(columns=1, rows=1),
        )
        fig.update_traces(hovertemplate='Dominio: %{customdata[0]:title} <br>Año: %{x} <br>Valor: %{y}<extra></extra>')

    elif type == "Gráfico de barras":
        data = data.sort_values(by='Diferencia', ascending=True)
        data["Color"] = np.where(data["Diferencia"]<0, 'red', 'green')

        fig = px.bar(data, y='Country', x='Diferencia', orientation='h',
                    hover_data=['Index Year'],
                    labels={'Country': 'País', 'Diferencia': 'Diferencia'},
                    height=700)

        fig.update_traces(
            marker_color=data["Color"],
            hovertemplate='País: %{y}</b><br>Diferencia en el índice: %{x}<br>Año: %{customdata[0]}<extra></extra>'
            
            )
        fig.update_layout(
            title=f'Evolución del dominio {d}',
            xaxis_title=f'Diferencia {data['Index Year'].values[0]}',
            yaxis_title='País',
            grid=dict(columns=1, rows=1)
        )

    elif type == "BarraEurope":
        df = data.melt(id_vars=['Country', 'code', 'Index Year'], var_name='Variable', value_name='Valor')
        df = df[df['Variable'].isin(variables)]

        fig = go.Figure()

        # Trazas para las líneas (diferencia entre Europa y el pais seleccionado)
        for i in sorted(df["Variable"].tolist(), reverse=True):

            aux = df[df["Variable"] == i]
            eu = aux[aux["Country"] == "Europe"]["Valor"].values[0]
            coun = aux[aux["Country"] != "Europe"]["Valor"].values[0]
            fig.add_trace(go.Scatter(
                x=[eu, coun],
                y=[i, i],
                mode='lines',
                line=dict(color='grey', width=1),
                showlegend=False,
                hovertemplate=None,
                hoverinfo='skip'
            ))

        # Trazas para los puntos (Europa y el pais seleccionado)
        for i in df["Country"].unique():
            val = df[df["Country"] == i]["Valor"].round(2).tolist()
            dom = df[df["Country"] == i]["Variable"].tolist()
            col = default_color if i != "Europe" else "blue"
            fig.add_trace(go.Scatter(
                y=dom,
                x=val,
                mode='markers',
                customdata=df[df["Country"] == i][["Country", "Variable"]].values,
                marker=dict(color=col, size=10),
                name=i,
                hovertemplate='<b>%{customdata[0]}</b><br>Dominio: %{customdata[1]}<br>Valor: %{x}<extra></extra>'
            ))

        country = df[df["Country"] != 'Europe']["Country"].values[0]
        fig.update_layout(
            title=f'Comparación de los dominios en {country} respecto a Europa',
            xaxis_title='Valor del dominio',
            yaxis_title='Dominio',
            legend_itemclick=False,
            legend_itemdoubleclick=False
        )



    elif type == "Proporción":

        data = data[data["Country"] != 'Europe']
        pos = data[data['Diferencia'] > 0].shape[0]
        neg = data[data['Diferencia'] < 0].shape[0]

        if pos == 0 and neg == 0:
            st.write(f"La diferencia es 0 en todos los países entre {data["Index Year"].values[0]}. No se puede realizar la comparación.")
            return

        else:
            # Pie chart
            fig = px.pie(values=[pos, neg], names=['mejora', 'empeoramiento'],
                        title=f'Proporción de países con mejora y empeoramiento <br>en el dominio {d}',
                        height=500, width=500,
                        color_discrete_sequence=['green', 'red'])
            fig.update_traces(
                hovertemplate='Países con %{label}: %{value} <br>%{percent:.2%}<extra></extra>',
                labels=['Mejora', 'Empeoramiento']
            )
            
            cols = st.columns(2)
            with cols[0]:
                st.plotly_chart(fig)
            with cols[1]:
                st.write(f"")
                st.write(f"")
                custom_metrics(data)

    elif type == "Boxplot":

        df = data.melt(id_vars=['Country', 'code', 'Index Year'], var_name='Variable', value_name='Value')
        df = df[df['Variable'].isin(variables)]

        fig = px.box(df, x='Variable', y='Value', color='Variable',
                    labels={'Variable': 'Dominio', 'Value': 'Valor'},
                    category_orders={'Variable': sorted(variables)},
                    title=f'Distribución de los dominios en {data["Country"].values[0]}',
                    height=700)
        fig.update_layout(
            xaxis_title='Dominio del dominio',
            yaxis_title='Valor',
            grid=dict(columns=1, rows=1)
        )

    elif type == "Mapa":
        custom_map(data, variables, "Diferencia", escala=escala, paleta="PiYG")

    elif type == "Cartograma":
        custom_cartogram(data, variables, "Diferencia", escala=escala, paleta="PiYG", pagina=pagina)

    if type not in ["Proporción", "Mapa", "Cartograma"]:
        st.plotly_chart(fig)

def custom_visualization(data, type_vis, variables, dominio, escala=False, auxiliar=None, paleta="BuPu", pagina=None):

    data = data.sort_values(by='Country', ascending=False)
    data["Type"] = dominio

    variable = data.columns[2]
    if type_vis == "Gráfico de barras" or type_vis == "Gráfico de líneas":
        st.title(f'Comparación del dominio {dominio} en {data["Index Year"].values[0]}')
        cols = st.columns(3)
        eu = data[data["Country"] == 'Europe'][dominio].values[0]
        with cols[0]:
            st.metric(label="Media en Europa", value=eu.round(2))
        with cols[1]:
            st.metric(label="Países por encima de la media", value=data[data[dominio] > eu].shape[0])
        with cols[2]:
            st.metric(label="Países por debajo de la media", value=data[data[dominio] < eu].shape[0])
        custom_bars(data, variables, dominio)
        
    elif type_vis == "Heatmap":
        st.title(f'Comparación del dominio {dominio}')
        # sns heatmap to plotly
        data = data.sort_values(by='code', ascending=True)
        data[dominio] = round(data[dominio], 2)
        fig = go.Figure(data=go.Heatmap(
            z=data[dominio],
            x=data['code'],
            y=data['Index Year'],
            customdata=data[['Country', 'Type']],
            colorscale='BuPu'))
        fig.update_layout(
            title=f'Comparación del dominio {dominio}',
            legend_title_text='Valor',
            xaxis_title='País',
            yaxis_title='Año',
            grid=dict(columns=1, rows=1)
        )
        fig.update_traces(hovertemplate='País: %{customdata[0]}<br>Año: %{y}<br>Dominio: %{customdata[1]}<br>Valor: %{z}<extra></extra>')
        fig.update_yaxes(type='category', categoryarray=sorted(data['Index Year'].unique()))
        st.plotly_chart(fig)

    elif type_vis == "Mapa":
        custom_map(data, variables, dominio, escala=escala, paleta=paleta, auxiliar=auxiliar)

    elif type_vis == "Cartograma":
        custom_cartogram(data, variables, dominio, escala=escala, paleta=paleta, auxiliar=auxiliar, pagina=pagina)
                         
def custom_bars(data, variable, dominio):
    if variable == "Año concreto":
        data = data.sort_values(by=dominio, ascending=True)
        fig = px.bar(data, y='Country', x=dominio, orientation='h',
                     hover_data=['Type'],
                     labels={'Country': 'País', dominio: 'Valor'},
                     height=700)
        # change color for Europe
        fig.update_traces(marker_color=np.where(data['Country'] == 'Europe', '#457b9d', default_color))
        fig.update_layout(
            title=f'Dominio {dominio} en {data["Index Year"].values[0]}',
            xaxis_title=dominio,
            yaxis_title='País',
            grid=dict(columns=1, rows=1)
        )
        fig.update_traces(hovertemplate='País: %{y}<br>Dominio: %{customdata[0]}<br>Valor: %{x}<extra></extra>')
    else:
        fig = px.line(data, x='Index Year', y=dominio, color='Country',
                        hover_data=['Type'],

                        labels={'Index Year': 'Año', dominio: 'Valor'},
                        height=700)
        fig.update_layout(
            title=f'Dominio {dominio}',
            xaxis_title='Año',
            yaxis_title='Valor',
            grid=dict(columns=1, rows=1)
        )
    st.plotly_chart(fig)

def custom_cartogram(data, variables, dominio, escala, paleta="BuPu", auxiliar=None, pagina=None):

    coords = load_coords()
    data = data.merge(coords, how='left', left_on='code', right_on='CNTR_CODE')
    data["lat"] = data["lat"].astype(int)
    data["lng"] = data["lng"].astype(int)
    data = data.sort_values(by='code', ascending=True)
    data[dominio] = round(data[dominio], 2)
    data["Type"] = dominio

    # Crear una figura para Plotly
    fig = go.Figure()

    # Crear una paleta de colores utilizando matplotlib
    cmap = plt.get_cmap(paleta)

    if "concreto" in variables and pagina == "Comparación":
        st.title(f'Comparación del dominio {dominio} en {data["Index Year"].values[0]}')
    elif pagina == "Comparación":
        st.title(f'Comparación del dominio {dominio}')
    
    if "concreto" in variables:
        # st.title(f'Comparación del dominio {dominio} en {data["Index Year"].values[0]}')

        if dominio == "Diferencia":
            vmax = data[dominio].abs().max()
            vmin = -vmax
        else:
            vmax = data[dominio].max()
            vmin = data[dominio].min()

        norm = Normalize(vmin=vmin, vmax=vmax)

        # Asignar color del texto basado en la luminancia del color de fondo
        data["color"] = data[dominio].apply(lambda x: get_color(x, cmap, norm))
        data["fillcolor"] = data["color"].apply(lambda x: x[0])
        data["rgb"] = data["color"].apply(lambda x: x[1])
        data["col"] = data["rgb"].apply(lambda x: get_text_color(x))

        for i, row in data.iterrows():
            vertices = square_vertices(row["lng"], row["lat"])
            x, y = zip(*vertices)
            x = list(x) + [x[0]]  # Cerrar el polígono
            y = list(y) + [y[0]]  # Cerrar el polígono
            # x_values.append(x)
            # y_values.append(y)

            if row["Country"] == "Europe":
                fig.add_trace(go.Scatter(
                    x = [row["lng"]], y = [row["lat"]], mode='markers', name="",
                    text = f'{row["Country"]} <br>Dominio: {row["Type"]}<br>Año: {row["Index Year"]} <br>Valor: {row[dominio]}',
                    hoverinfo='text', showlegend=False,
                ))
                fig.update_traces(marker=dict(color=row["fillcolor"], size=30))

            else:
                fig.add_trace(go.Scatter(
                x=x, y=y, mode='lines', name="",
                text = f'Pais: {row["Country"]} <br>Dominio: {row["Type"]}<br>Año: {row["Index Year"]} <br>Valor: {row[dominio]}',
                hoverinfo='text',
                line=dict(color=row["fillcolor"], width=0),
                customdata=[[row["Country"], row["Type"], row[dominio]]],
                fill='toself', showlegend=False, 
                
                fillcolor=row["fillcolor"]
            ))
            # Añadir anotaciones
            fig.add_annotation(
                x=row["lng"], y=row["lat"], text=row["code"],
                font=dict(color=row["col"]),
                showarrow=False
            )

        if dominio != "Diferencia":
            fig.update_layout(
                title=f'Comparación del dominio {dominio}',
            )
        

        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            text=[None],
            mode='markers',
            showlegend=False,
            marker=dict(
                colorscale=paleta,
                cmin=vmin,
                cmax=vmax,
                colorbar=dict(
                    title='',
                    tickvals=np.linspace(vmin, vmax, 7),
                    ticktext=[str(round(i, 2)) for i in np.linspace(vmin, vmax, 7)],
                    yanchor="top", y=1, x=1.1,
                    ticks="outside"
                ),
            ),
            hoverinfo='skip'
        ))

        fig.update_layout(
            template='simple_white',
            height=580, width=580, 
            xaxis = None, yaxis = None, xaxis_showgrid=False, yaxis_showgrid=False,
            margin=dict(l=0, r=0, t=25, b=0)
        )
        fig.update_xaxes(ticks='', showgrid=False, showline=False, zeroline=False, visible =False)
        fig.update_yaxes(ticks='', showgrid=False,  showline=False, zeroline=False, visible =False)

        st.plotly_chart(fig)

    else:
        # st.title(f'Comparación del dominio {dominio}')
        years, _ = get_years()
        cols = st.columns(3)

        print(data)

        if dominio == "Diferencia":
            values = data["Dominio"].unique()
            t = 'Dominio'
        else:
            values = sorted(data['Index Year'].unique())
            t = 'Index Year'

        for ind, year in enumerate(values):
            indice = ind % 3

            df = data[data[t] == year].reset_index(drop=True)

            if dominio == "Diferencia" and escala == False:
                limites = [-df[dominio].abs().max(), df[dominio].abs().max()]

            elif dominio == "Diferencia":
                limites = [-data["Diferencia"].abs().max(), data["Diferencia"].abs().max()]

            elif escala and auxiliar is None:
                limites = [data[dominio].min(), data[dominio].max()]
            elif escala == False and auxiliar is None:
                limites = [df[dominio].min(), df[dominio].max()]
            elif escala == False:
                df1 = auxiliar[auxiliar[t] == year]
                limites = [df1[dominio].min(), df1[dominio].max()]
            else:
                limites = [auxiliar[dominio].min(), auxiliar[dominio].max()]

            fig = go.Figure()

            norm = Normalize(vmin=limites[0], vmax=limites[1])

            # Asignar color del texto basado en la luminancia del color de fondo
            df["color"] = df[dominio].apply(lambda x: get_color(x, cmap, norm))
            df["fillcolor"] = df["color"].apply(lambda x: x[0])
            df["rgb"] = df["color"].apply(lambda x: x[1])
            df["col"] = df["rgb"].apply(lambda x: get_text_color(x))

            for i, row in df.iterrows():
                vertices = square_vertices(row["lng"], row["lat"])
                x, y = zip(*vertices)
                x = list(x) + [x[0]]  # Cerrar el polígono
                y = list(y) + [y[0]]  # Cerrar el polígono
                # x_values.append(x)
                # y_values.append(y)

                if row["Country"] == "Europe":
                    fig.add_trace(go.Scatter(
                        x = [row["lng"]], y = [row["lat"]], mode='markers', name="",
                        text = f'{row["Country"]} <br>Dominio: {row["Type"]}<br>Año: {row["Index Year"]} <br>Valor: {row[dominio]}',
                        hoverinfo='text', showlegend=False,
                    ))
                    fig.update_traces(marker=dict(color=row["fillcolor"], size=30))

                else:
                    fig.add_trace(go.Scatter(
                        x=x, y=y, mode='lines', name="",
                        text = f'Pais: {row["Country"]} <br>Dominio: {row["Type"]}<br>Año: {row["Index Year"]} <br>Valor: {row[dominio]}',
                        hoverinfo='text',
                        line=dict(color=row["fillcolor"], width=0),
                        customdata=[[row["Country"], row["Type"], row[dominio]]],
                        fill='toself', showlegend=False, 
                        
                        fillcolor=row["fillcolor"]
                    ))
                # Añadir anotaciones
                fig.add_annotation(
                    x=row["lng"], y=row["lat"], text=row["code"],
                    font=dict(color=row["col"]),
                    showarrow=False
                )

            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                text=[None],
                mode='markers',
                showlegend=False,
                marker=dict(
                    colorscale=paleta,
                    cmin=limites[0],
                    cmax=limites[1],
                    
                    colorbar=dict(
                        # title='Índice',
                        tickvals=np.linspace(limites[0], limites[1], 7),
                        ticktext=[str(round(i, 2)) for i in np.linspace(limites[0], limites[1], 7)],
                        yanchor="top", y=1, x=1.1,
                        thickness=15,
                        ticks="outside"
                    ),
                ),
                hoverinfo='skip'
            ))
            fig.update_layout(
                title=f'{year}',
                template='simple_white',
                height=400, width=400, 
                xaxis = None, yaxis = None, xaxis_showgrid=False, yaxis_showgrid=False,
                margin=dict(l=12.5, r=12.5, t=25, b=0)
                

            )
            fig.update_xaxes(ticks='', showgrid=False, showline=False, zeroline=False, visible =False)
            fig.update_yaxes(ticks='', showgrid=False,  showline=False, zeroline=False, visible =False)
            with cols[indice]:
                st.plotly_chart(fig)
                        


def custom_map(data, variables, dominio, escala, paleta="BuPu", auxiliar=None):

    if paleta != "PiYG":
        st.title(f'Comparación del dominio {dominio}')
    data = data.sort_values(by='Country', ascending=False)

    carto = load_cartography()
    carto["CNTR_CODE"] = carto["CNTR_CODE"].str.replace('EL', 'GR')
    if "concreto" in variables:
        # Crear un mapa de folium
        m = folium.Map(location=[55.0, 20], zoom_start=3.4, min_zoom=3,max_bounds=True,
                       min_lat=20, max_lat=75, min_lon=-28, max_lon=40)
        if type(data["Index Year"].values[0]) == str:
            vmax = data[dominio].abs().max()
            vmin = -vmax
        else:
            vmax = data[dominio].max()
            vmin = data[dominio].min()

        choropleth = folium.Choropleth(
            geo_data=carto,
            name='choropleth',
            data=data,
            columns=['code', dominio],
            key_on='feature.properties.CNTR_CODE',
            fill_color=paleta,
            nan_fill_color="gray",
            fill_opacity=1,
            threshold_scale = np.linspace(vmin, vmax, 7),
            line_opacity=0.2,
            legend_name=dominio,
            highlight=True,
        ).add_to(m)
        

        df = data.copy()
        year = df['Index Year'].values[0]
        for feature in choropleth.geojson.data['features']:
            country = feature['properties']['CNTR_CODE']

            feature['properties']['name'] = df[df['code'] == country].reset_index(drop=True).loc[0, "Country"] if country in df['code'].values else feature['properties']['Eng']
            feature['properties']['value'] = round(df[df['code'] == country].reset_index(drop=True).loc[0, dominio], 2) if country in df['code'].values else 0
            feature['properties']['year'] = str(year)
            
        
        choropleth.geojson.add_child(
            folium.features.GeoJsonTooltip(['name', 'year','value'], aliases=['Country', 'Year', dominio])
        )
        folium_static(m, width=800, height=500)
        # st.write(streamlit_js_eval(js_expressions='zoom', key=f'zoom',  want_output = True))
    else:
        make_map_responsive= """
            <style>
            iframe {
                width: 100%;
                
                height: 100%:
            }
            </style>
        """
        st.markdown(make_map_responsive, unsafe_allow_html=True)
        years , _ = get_years()
        cols = st.columns(3)
        #width = streamlit_js_eval(js_expressions='window.innerWidth', key=f'WIDTH',  want_output = True)
        height = 500

        if dominio == "Diferencia":
            values = sorted(data["Dominio"].unique())
            t = 'Dominio'
        else:
            values = sorted(data['Index Year'].unique())
            t = 'Index Year'

        for ind, year in enumerate(values):
            indice = ind % 3

            df = data[data[t] == year].reset_index(drop=True)

            if dominio == "Diferencia" and not escala:
                limites = [-df[dominio].abs().max(), df[dominio].abs().max()]   
            elif dominio == "Diferencia":
                limites = [-data[dominio].abs().max(), data[dominio].abs().max()]        
            elif escala and auxiliar is None:
                limites = [data[dominio].min(), data[dominio].max()]
            elif escala == False and auxiliar is None:
                limites = [df[dominio].min(), df[dominio].max()]
            elif escala == False:
                df1 = auxiliar[auxiliar['Index Year'] == year]
                limites = [df1[dominio].min(), df1[dominio].max()]
            else:
                limites = [auxiliar[dominio].min(), auxiliar[dominio].max()]


            if data["Country"].unique().shape[0] > 2:
                loc = [55, 15]
                m = folium.Map(location=loc, zoom_start=2.5, min_zoom=2.5,
                                zoom_control=False, max_bounds=True, min_lat=30, max_lat=75, min_lon=-40, max_lon=50)
            else:
                c = data.loc[data["Country"] != "Europe", "code"].values[0]
                c = carto[carto["CNTR_CODE"] == c]["geometry"].values[0].centroid
                loc = [c.y, c.x]
                m = folium.Map(location=loc, zoom_start=4, min_zoom=4, max_bounds=True, min_lat=30, max_lat=75, min_lon=-40, max_lon=50)

            choropleth = folium.Choropleth(
                geo_data=carto,
                name='choropleth',
                data=df,
                columns=['code', dominio],
                key_on='feature.properties.CNTR_CODE',
                fill_color=paleta,
                nan_fill_color="gray",
                fill_opacity=1,
                threshold_scale = np.linspace(limites[0], limites[1], 7),
                line_opacity=0.2,
                legend_name='',
                highlight=True
            )
            
            choropleth.add_to(m)
            choropleth.color_scale.width = 350

            for feature in choropleth.geojson.data['features']:
                country = feature['properties']['CNTR_CODE']

                feature['properties']['name'] = df[df['code'] == country].reset_index(drop=True).loc[0, "Country"] if country in df['code'].values else feature['properties']['Eng']
                feature['properties']['value'] = round(df[df['code'] == country].reset_index(drop=True).loc[0, dominio], 1) if country in df['code'].values else 0

                if dominio == "Diferencia":
                    feature['properties']['year'] = df["Index Year"].values[0]
                else:
                    feature['properties']['year'] = year.astype(str)
                
            
            choropleth.geojson.add_child(
                folium.features.GeoJsonTooltip(['name', 'year','value'], aliases=['Country', 'Year', dominio])
            )
            
            with cols[indice]:
                # write the year centered
                europa = df.loc[df['Country'] == 'Europe', dominio].values[0].round(2)
                st.subheader(f"{year}")
                st.write(f"Media en Europa: {europa}", unsafe_allow_html=True)
                folium_static(m)

def animation(data, dominio, tipo):

    if tipo == "Mapa":
        data = data[data['code'] != 'EU']
    gdf = load_cartography()
    gdf.index = gdf['CNTR_CODE']
    gdf = json.loads(gdf.to_json())

    vmin = data[dominio].min()
    vmax = data[dominio].max()

    data["Type"] = dominio

    if tipo == "Mapa":
    
        fig = px.choropleth_mapbox(data_frame=data,
                                geojson=gdf,
                                locations=data.code,
                                color=dominio,
                                center={'lat':55, 'lon':15},
                                mapbox_style='open-street-map',
                                zoom=2.5,
                                range_color=[vmin, vmax],
                                color_continuous_scale='BuPu',
                                animation_frame='Index Year',
                                height=650,
                                width=650,
                                custom_data=['Country', 'Index Year', "Type"],
                                hover_data={'Country': True, 'Index Year': True, dominio: True, 'code': False},
                                labels={dominio: dominio, 'Country': 'País', 'Index Year': 'Año'},
                                title=f'Evolución del dominio {dominio}',
                                )
        
        fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
        fig.update_layout( coloraxis_colorbar=dict(
                title="",
            )
        )
        st.plotly_chart(fig)

    else:
        animation_cartogram(data, dominio, vmin, vmax)


def animation_cartogram(data, dominio, vmin, vmax):

    coords = load_coords()
    data = data.merge(coords, how='left', left_on='code', right_on='CNTR_CODE')
    data["lat"] = data["lat"].astype(int)
    data["lng"] = data["lng"].astype(int)
    data = data.sort_values(by='code', ascending=True)
    data[dominio] = round(data[dominio], 2)
    data["Type"] = dominio

    years = sorted(data['Index Year'].unique())

    # make list of continents
    continents = []
    for continent in data["Country"]:
        if continent not in continents:
            continents.append(continent)
    # make figure
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }
    fig_dict["layout"]["hovermode"] = "closest"
    # fill in most of layout
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 500, "redraw": False},
                                    "fromcurrent": True, 
                                    # "transition": {"duration": 300,
                                    #                                     "easing": "quadratic-in-out"}
                                                                        }],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Año: ",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    # create color scale
    cmap = plt.get_cmap("BuPu")
    norm = Normalize(vmin=vmin, vmax=vmax)

    data["color"] = data[dominio].apply(lambda x: get_color(x, cmap, norm))
    data["fillcolor"] = data["color"].apply(lambda x: x[0])
    data["rgb"] = data["color"].apply(lambda x: x[1])
    data["col"] = data["rgb"].apply(lambda x: get_text_color(x))

    # make data
    year = sorted(data['Index Year'].unique())[0]
    for continent in continents:
        dataset_by_year = data[data["Index Year"] == year]
        dataset_by_year_and_cont = dataset_by_year[
            dataset_by_year["Country"] == continent]
        
        vertices = square_vertices(dataset_by_year_and_cont["lng"].values[0], dataset_by_year_and_cont["lat"].values[0])
        x, y = zip(*vertices)
        x = list(x) + [x[0]]  # Cerrar el polígono
        y = list(y) + [y[0]] 

        count = dataset_by_year_and_cont["Country"].values[0]
        dom = dataset_by_year_and_cont["Type"].values[0]
        val = dataset_by_year_and_cont[dominio].values[0]
        year = dataset_by_year_and_cont["Index Year"].values[0]

        data_dict = {
            "x": x,
            "y": y,
            "mode": "lines",
            "text": f'Pais: {count} <br>Dominio: {dom}<br>Año: {year} <br>Valor: {val}',
            "hoverinfo": 'text',
            "line": {"color": dataset_by_year_and_cont["fillcolor"].values[0], "width": 0},
            "customdata": [[count, dom, val]],
            "fill": 'toself',
            "showlegend": False,
            "fillcolor": dataset_by_year_and_cont["fillcolor"].values[0],
            'name': ''

        }
        fig_dict["data"].append(data_dict)

        data_dict = {
            "x": [dataset_by_year_and_cont["lng"].values[0]],
            "y": [dataset_by_year_and_cont["lat"].values[0]],
            "mode": "text",
            "text": dataset_by_year_and_cont["code"].values[0],
            'textfont': {'color': 'black'},
            'showlegend': False
        }
        fig_dict["data"].append(data_dict)


    # make frames
    for year in years:
        frame = {"data": [], "name": str(year)}
        
        for continent in continents:
            dataset_by_year = data[data["Index Year"] == year]
            dataset_by_year_and_cont = dataset_by_year[
                dataset_by_year["Country"] == continent]
            
            vertices = square_vertices(dataset_by_year_and_cont["lng"].values[0], dataset_by_year_and_cont["lat"].values[0])
            x, y = zip(*vertices)
            x = list(x) + [x[0]]  # Cerrar el polígono
            y = list(y) + [y[0]] 

            count = dataset_by_year_and_cont["Country"].values[0]
            dom = dataset_by_year_and_cont["Type"].values[0]
            val = dataset_by_year_and_cont[dominio].values[0]
            year = dataset_by_year_and_cont["Index Year"].values[0]

            data_dict = {
                "x": x,
                "y": y,
                "mode": "lines",
                "text": f'Pais: {count} <br>Dominio: {dom}<br>Año: {year} <br>Valor: {val}',
                "hoverinfo": 'text',
                "line": {"color": dataset_by_year_and_cont["fillcolor"].values[0], "width": 0},
                "customdata": [[count, dom, val]],
                "fill": 'toself',
                "showlegend": False,
                "fillcolor": dataset_by_year_and_cont["fillcolor"].values[0],
                'name': ''

            }
            frame["data"].append(data_dict)

            data_dict = {
                "x": [dataset_by_year_and_cont["lng"].values[0]],
                "y": [dataset_by_year_and_cont["lat"].values[0]],
                "mode": "text",
                "text": dataset_by_year_and_cont["code"].values[0],
                'textfont': {'color': 'black'},
                'showlegend': False,
                'hoverinfo': 'none'
            }
            frame["data"].append(data_dict)


        fig_dict["frames"].append(frame)
        slider_step = {"args": [
            [year],
            {
                "frame": {"duration": 300, "redraw": False},
            "mode": "immediate",
            "transition": {"duration": 300}}
        ],
            "label": str(year),
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)


    fig_dict["layout"]["sliders"] = [sliders_dict]


    fig = go.Figure(fig_dict)

    fig.update_layout(
        title=f'Evolución del dominio {dominio}',
        height=650, width=650, 
        xaxis = None, yaxis = None, xaxis_showgrid=False, yaxis_showgrid=False,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    # add colorbar
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        text=[None],
        opacity=0,
        mode='markers',
        showlegend=False,
        marker=dict(
            colorscale='BuPu',
            cmin=vmin,
            cmax=vmax,
            colorbar=dict(
                title='',
                tickvals=np.linspace(vmin, vmax, 7),
                ticktext=[str(round(i, 2)) for i in np.linspace(vmin, vmax, 7)],
                yanchor="top", y=1, x=1.1,
                thickness=15,
                ticks="outside"
            ),
        ),
        hoverinfo='skip',
        visible=True
    ))

    fig.update_xaxes(ticks='', showgrid=False, showline=False, zeroline=False, visible =False)
    fig.update_yaxes(ticks='', showgrid=False,  showline=False, zeroline=False, visible =False)

    st.plotly_chart(fig)