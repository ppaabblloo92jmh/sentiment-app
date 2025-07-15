
import streamlit as st
from sentiment import SentimentAnalyzer

st.title("🧠 Análisis de Sentimiento Financiero")
analyzer = SentimentAnalyzer()

option = st.selectbox("Elige una opción:", ["Texto personalizado", "Sentimiento del mercado"])

if option == "Texto personalizado":
    text = st.text_area("Introduce un texto para analizar:")
    if st.button("Analizar Sentimiento"):
        if text:
            result = analyzer.analyze_text(text)
            st.write("Resultado:", result)
elif option == "Sentimiento del mercado":
    if st.button("Obtener Sentimiento"):
        result = analyzer.get_market_sentiment()
        st.write("Sentimiento del mercado:", result)
