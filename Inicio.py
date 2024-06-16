# Librerias
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title='Índice de Igualdad de Género de la UE', layout='wide',
                   initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

# Title
st.title('Índice de Igualdad de Género de la UE')

st.markdown('''
    El **Índice de Igualdad de Género** es una medida que evalúa la igualdad de género en los países de la Unión Europea que otorga a la UE y a los Estados miembros una puntuación del 1 al 100. 

    - Una puntuación de 100 significaría que un país ha alcanzado la plena igualdad entre mujeres y hombres.
                
    - Este índice se basa en la igualdad de género en seis dimensiones: trabajo, dinero, conocimiento, tiempo, poder y salud.
            
    Si deseas obtener más información sobre el índice, puedes acceder a la [web oficial](https://eige.europa.eu/gender-equality-index/2022/EU) y descargar los datos desde [aquí](https://eige.europa.eu/modules/custom/eige_gei/app/content/downloads/gender-equality-index-2013-2015-2017-2019-2020-2021-2022-2023.xlsx).
    
    ---
            
    Esta web ha sido creada para analizar los datos del índice de igualdad de género de la UE como proyecto final de la asignatura de Visualización de Datos del Máster Universitario en Inteligencia Artificial, Reconocimiento de Formas e Imagen Digital (MUIARFID) de la Universitat Politècnica de València durante el curso 2023-2024.
            
    Para acceder al vídeo de presentación de la web, haz clic [aquí](https://media.upv.es/player/?id=8fce02e0-2bb6-11ef-b5d5-b350270db618).
                
    '''
)



