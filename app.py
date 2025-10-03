import pandas as pd
import streamlit as st

from data_agent import DataExplorerAgent
from kpi_analyzer import KPIAnalyzer, KPIMetric

st.set_page_config(page_title="Dataset Previewer", layout="wide")
st.title("📊 Dataset Previewer")


def load_dataset(upload) -> pd.DataFrame | None:
    file_name = upload.name.lower()
    try:
        if file_name.endswith(".csv"):
            df = pd.read_csv(upload)
        elif file_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(upload)
        elif file_name.endswith(".parquet"):
            df = pd.read_parquet(upload)
        elif file_name.endswith(".json"):
            df = pd.read_json(upload)
        else:
            st.error("❌ Format non supporté.")
            return None
    except Exception as exc:
        st.error(f"Erreur lors de la lecture du fichier : {exc}")
        return None

    return _auto_cast_columns(df)


def _auto_cast_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_casted = df.copy()
    datetime_keywords = [
        "date",
        "time",
        "timestamp",
        "jour",
        "semaine",
        "mois",
        "annee",
        "year",
    ]
    for column in df_casted.columns:
        if df_casted[column].dtype == object:
            lower_name = column.lower()
            if any(keyword in lower_name for keyword in datetime_keywords):
                converted = pd.to_datetime(df_casted[column], errors="coerce")
                if converted.notna().sum() > 0:
                    df_casted[column] = converted
            else:
                # Try casting numeric-looking object columns
                converted = pd.to_numeric(df_casted[column], errors="coerce")
                if converted.notna().sum() > 0:
                    df_casted[column] = converted
    return df_casted


def show_overview_page(df: pd.DataFrame) -> None:
    st.subheader("📄 Aperçu des données")
    st.dataframe(df, use_container_width=True)

    st.markdown("### Types de colonnes")
    st.write(df.dtypes)

    if st.checkbox("📈 Afficher les statistiques descriptives"):
        st.write(df.describe(include="all"))


def _render_metric_card(container, metric: KPIMetric) -> None:
    container.markdown(f"**{metric.label}**")
    container.markdown(
        f"<span style='font-size:1.8rem;font-weight:600'>{metric.value}</span>",
        unsafe_allow_html=True,
    )
    if metric.description:
        container.caption(metric.description)
    if metric.column:
        container.caption(f"Colonne source : `{metric.column}`")


def show_kpi_page(df: pd.DataFrame) -> None:
    st.subheader("🎯 Explorateur de KPI")
    analyzer = KPIAnalyzer(df)
    metrics, details = analyzer.suggest()

    if not metrics:
        st.info("Aucun KPI pertinent n'a été détecté automatiquement.")
        return

    st.markdown("### KPI suggérés")
    cards_per_row = 3
    for start in range(0, len(metrics), cards_per_row):
        row_metrics = metrics[start : start + cards_per_row]
        columns = st.columns(len(row_metrics))
        for container, metric in zip(columns, row_metrics):
            _render_metric_card(container, metric)

    if details:
        st.markdown("### Analyses complémentaires")
    for detail in details:
        st.markdown(f"#### {detail.title}")
        if detail.description:
            st.caption(detail.description)
        st.dataframe(detail.dataframe, use_container_width=True)


def show_advanced_page(df: pd.DataFrame) -> None:
    st.subheader("🔍 Analyse avancée")
    agent = DataExplorerAgent(df)

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
        n_clusters = st.number_input("Nombre de clusters", min_value=2, max_value=10, value=3)
        labels = agent.cluster(int(n_clusters))
        df_clustered = df.copy()
        df_clustered["cluster"] = labels
        st.write(df_clustered)


if "df" not in st.session_state:
    st.session_state["df"] = None
    st.session_state["file_name"] = None

uploaded_file = st.file_uploader(
    "📁 Upload your dataset", type=["csv", "xlsx", "xls", "parquet", "json"]
)

dataset = st.session_state.get("df")
if uploaded_file is not None:
    dataset = load_dataset(uploaded_file)
    if dataset is not None:
        st.session_state["df"] = dataset
        st.session_state["file_name"] = uploaded_file.name
        st.success(
            f"✅ Fichier {uploaded_file.name} chargé avec succès ! Dimensions : "
            f"{dataset.shape[0]} lignes × {dataset.shape[1]} colonnes"
        )
    else:
        dataset = st.session_state.get("df")

if st.session_state.get("df") is None:
    st.info("Veuillez uploader un fichier CSV, Excel, JSON ou Parquet.")
    st.stop()

st.sidebar.header("Navigation")
selected_page = st.sidebar.radio(
    "Aller à", ("Aperçu des données", "Explorateur de KPI", "Analyse avancée")
)

file_name = st.session_state.get("file_name")
if file_name:
    st.caption(f"Fichier actuellement chargé : **{file_name}**")

dataset = st.session_state["df"]

if selected_page == "Aperçu des données":
    show_overview_page(dataset)
elif selected_page == "Explorateur de KPI":
    show_kpi_page(dataset)
else:
    show_advanced_page(dataset)
