import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px

st.markdown("""
    <style>
    .stApp {
        background-color: 	#66b2b2;
        color: white;
    }
    .stButton>button {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# el titulo y la descripción
st.title('🔍 Análisis de Sentimientos de Comentarios de Instagram y Facebook📊')
st.markdown("<h3 style='text-align: center;'>Sube tus archivos CSV y analiza los sentimientos de las publicaciones</h3>", unsafe_allow_html=True)

# Subir los archivos instagram y facebook
st.sidebar.header('Sube tus archivos CSV')
archivo_ig = st.sidebar.file_uploader("Comentarios de Instagram", type=["csv"])
archivo_face = st.sidebar.file_uploader("Comentarios de Facebook", type=["csv"])

# Subir una imagen
imagenes = st.sidebar.file_uploader("Sube una o más imágenes", accept_multiple_files=True, type=["jpg", "png"])

if imagenes:
    for uploaded_image in imagenes:
        st.image(uploaded_image, caption='Imagen subida por el usuario', use_column_width=True)


analizador = SentimentIntensityAnalyzer()

def obtener_sentimiento_vader(comentario):
    if isinstance(comentario, str):
        analisis = analizador.polarity_scores(comentario)
        return analisis['compound']
    else:
        return None

def clasificar_sentimiento(polaridad):
    if polaridad is not None:
        if polaridad > 0.1:
            return "positivo"
        elif polaridad < -0.1:
            return "negativo"
    return "neutral"

def analizar_sentimientos(df, plataforma):
    st.subheader(f'Comentarios de {plataforma}')
    st.write(df.head())

    df['Sentimiento'] = df['Comment'].apply(obtener_sentimiento_vader)
    df['Sentimiento'] = pd.to_numeric(df['Sentimiento'], errors='coerce')
    df.dropna(subset=['Sentimiento'], inplace=True)
    df['Clasificación Sentimiento'] = df['Sentimiento'].apply(clasificar_sentimiento)
    
    sentimiento_counts = df['Clasificación Sentimiento'].value_counts().reset_index()
    sentimiento_counts.columns = ['Sentimiento', 'Count']
    
    fig_sentimiento = px.bar(sentimiento_counts, x='Sentimiento', y='Count', title=f'Distribución de Sentimientos en {plataforma}')
    st.plotly_chart(fig_sentimiento)

    st.subheader(f'Informe de Patrones de Sentimientos en {plataforma}')
    st.write(f"**Total de Comentarios Analizados:** {len(df)}")
    st.write(f"**Distribución de Sentimientos:**")
    st.write(sentimiento_counts)

if archivo_ig is not None:
    comentarios_ig = pd.read_csv(archivo_ig)
    analizar_sentimientos(comentarios_ig, 'Instagram')

if archivo_face is not None:
    comentarios_face = pd.read_csv(archivo_face)
    analizar_sentimientos(comentarios_face, 'Facebook')

