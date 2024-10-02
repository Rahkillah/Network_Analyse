import streamlit as st

st.title("Analyse INTRUSIOIN")

file_upload = st.file_uploader("Choisier le fichier de test", type={'csv'})

if file_upload:
    st.download_button("Alaina le fichier", file_upload, "lefichier.csv", mime="file/csv")