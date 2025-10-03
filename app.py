import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data_agent import DataExplorerAgent

st.set_page_config(page_title="Dataset Previewer", layout="wide")
st.title("📊 Dataset Previewer")

uploaded_file = st.file_uploader(
    "📁 Upload your dataset", type=["csv", "xlsx", "xls", "parquet", "json"]
)

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
            st.success(
                f"✅ Fichier chargé avec succès ! Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes"
            )
            st.dataframe(df, use_container_width=True)
            st.markdown("### Aperçu rapide des colonnes")
            st.write(df.dtypes)
            if st.checkbox("📈 Afficher les statistiques descriptives"):
                st.write(df.describe(include="all"))

            # AI agent features
            agent = DataExplorerAgent(df)
            st.markdown("---")
            st.header("🔍 Analyse avancée")
            col1, col2 = st.columns(2)
            with col1:
                if st.checkbox("Afficher la heatmap de corrélation"):
                    heatmap = agent.correlation_heatmap()
                    st.image(heatmap)
            with col2:
                if st.checkbox("Détecter les valeurs aberrantes"):
                    outliers = agent.detect_outliers()
                    if outliers.empty:
                        st.info("Aucune valeur aberrante détectée")
                    else:
                        st.write(outliers)

            if st.checkbox("Clusteriser les données"):
                n_clusters = st.number_input(
                    "Nombre de clusters", min_value=2, max_value=10, value=3
                )
                labels = agent.cluster(n_clusters)
                df_clustered = df.copy()
                df_clustered["cluster"] = labels
                st.write(df_clustered)

            # ✅ Exporter les noms de colonnes
            st.markdown("---")
            st.subheader("⬇️ Exporter les noms des colonnes")
            if st.button("Télécharger la liste des colonnes"):
                column_names = "\n".join(df.columns)
                st.download_button(
                    label="📥 Télécharger colonnes.txt",
                    data=column_names,
                    file_name="colonnes.txt",
                    mime="text/plain",
                )

    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
else:
    st.info("Veuillez uploader un fichier CSV, Excel, JSON ou Parquet.")
