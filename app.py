import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="Dataset Previewer", layout="wide")
st.title("📊 Dataset Previewer")

uploaded_file = st.file_uploader("📁 Upload your dataset", type=["csv", "xlsx", "xls", "parquet", "json"])

if uploaded_file:
    file_name = uploaded_file.name.lower()
    try:
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
            df = pd.read_excel(uploaded_file)
        elif file_name.endswith(".parquet"):
            df = pd.read_parquet(uploaded_file)
        elif file_name.endswith(".json"):
            df = pd.read_json(uploaded_file)
        else:
            st.error("❌ Format non supporté.")
            df = None

        if df is not None:
            st.success(f"✅ Fichier chargé avec succès ! Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")
            st.dataframe(df, use_container_width=True)
            st.markdown("### Aperçu rapide des colonnes")
            st.write(df.dtypes)
            if st.checkbox("📈 Afficher les statistiques descriptives"):
                st.write(df.describe(include="all"))

    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
else:
    st.info("Veuillez uploader un fichier CSV, Excel, JSON ou Parquet.")
