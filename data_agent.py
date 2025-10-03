import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from io import BytesIO

class DataExplorerAgent:
    """Simple agent for automated data exploration."""

    def __init__(self, df: pd.DataFrame):
        # Only keep numeric columns for ML tasks
        self.df = df.select_dtypes(include='number')

    def correlation_heatmap(self) -> BytesIO:
        """Return a buffer containing the correlation heatmap."""
        corr = self.df.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, ax=ax, annot=True, cmap='coolwarm', fmt='.2f')
        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format="png")
        plt.close(fig)
        buffer.seek(0)
        return buffer

    def detect_outliers(self, contamination: float = 0.05) -> pd.DataFrame:
        """Detect outliers using IsolationForest."""
        if self.df.empty:
            return pd.DataFrame()
        model = IsolationForest(contamination=contamination, random_state=42)
        numeric_data = self.df.fillna(0)
        model.fit(numeric_data)
        preds = model.predict(numeric_data)
        outliers = self.df[preds == -1]
        return outliers

    def cluster(self, n_clusters: int = 3) -> pd.Series:
        """Cluster data using KMeans and return cluster labels."""
        if self.df.empty:
            return pd.Series(dtype=int)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        data = self.df.fillna(0)
        labels = kmeans.fit_predict(data)
        return pd.Series(labels, index=self.df.index)
