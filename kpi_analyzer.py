from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import pandas as pd


@dataclass
class KPIMetric:
    label: str
    value: str
    column: Optional[str]
    description: str


@dataclass
class KPIDetail:
    title: str
    dataframe: pd.DataFrame
    description: Optional[str] = None


def _format_number(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return "N/A"

    if abs(value) >= 100:
        formatted = f"{value:,.0f}"
    else:
        formatted = f"{value:,.{decimals}f}"

    return formatted.replace(",", "\u202f")


def _format_percent(value: float, decimals: int = 1) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}%}"


class KPIAnalyzer:
    """Generate KPI suggestions based on dataset column names."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_df = df.select_dtypes(include=["number"])
        self.categorical_df = df.select_dtypes(include=["object", "category"])

    def suggest(self) -> Tuple[List[KPIMetric], List[KPIDetail]]:
        metrics: List[KPIMetric] = []
        details: List[KPIDetail] = []

        metrics.extend(self._base_metrics())
        metrics.extend(self._numeric_keyword_metrics())
        metrics.extend(self._categorical_metrics())
        metrics.extend(self._datetime_metrics())

        details.extend(self._top_categories_details())

        # Deduplicate metrics by (label, column)
        seen = set()
        unique_metrics: List[KPIMetric] = []
        for metric in metrics:
            key = (metric.label, metric.column)
            if key in seen:
                continue
            seen.add(key)
            unique_metrics.append(metric)

        return unique_metrics, details

    def _base_metrics(self) -> Iterable[KPIMetric]:
        rows, cols = self.df.shape
        return [
            KPIMetric(
                label="Lignes disponibles",
                value=_format_number(rows, decimals=0),
                column=None,
                description="Nombre total d'enregistrements dans le fichier.",
            ),
            KPIMetric(
                label="Colonnes disponibles",
                value=_format_number(cols, decimals=0),
                column=None,
                description="Nombre total de colonnes détectées.",
            ),
        ]

    def _numeric_keyword_metrics(self) -> Iterable[KPIMetric]:
        if self.numeric_df.empty:
            return []

        rules = [
            {
                "keywords": ["revenue", "revenu", "sales", "vente", "turnover", "income", "ca"],
                "label": "Revenu total",
                "agg": lambda s: s.sum(),
                "description": "Somme totale des montants de la colonne {column}.",
            },
            {
                "keywords": ["profit", "benef", "margin"],
                "label": "Profit total",
                "agg": lambda s: s.sum(),
                "description": "Bénéfice cumulé calculé sur la colonne {column}.",
            },
            {
                "keywords": ["cost", "cout", "expense", "depense"],
                "label": "Coût total",
                "agg": lambda s: s.sum(),
                "description": "Somme totale des coûts basés sur {column}.",
            },
            {
                "keywords": ["price", "prix", "tarif", "amount", "montant"],
                "label": "Valeur moyenne",
                "agg": lambda s: s.mean(),
                "description": "Valeur moyenne observée dans la colonne {column}.",
            },
            {
                "keywords": ["quantity", "quantite", "qty", "volume"],
                "label": "Quantité totale",
                "agg": lambda s: s.sum(),
                "description": "Volume total agrégé pour la colonne {column}.",
            },
            {
                "keywords": ["rate", "taux", "ratio", "pourcentage"],
                "label": "Taux moyen",
                "agg": lambda s: s.mean(),
                "description": "Taux moyen calculé pour la colonne {column}.",
                "formatter": _format_percent,
            },
            {
                "keywords": ["score", "note", "evaluation"],
                "label": "Score moyen",
                "agg": lambda s: s.mean(),
                "description": "Score moyen observé dans la colonne {column}.",
            },
            {
                "keywords": ["age"],
                "label": "Âge moyen",
                "agg": lambda s: s.mean(),
                "description": "Âge moyen calculé sur la colonne {column}.",
            },
            {
                "keywords": ["duree", "duration", "temps", "time"],
                "label": "Durée moyenne",
                "agg": lambda s: s.mean(),
                "description": "Durée moyenne mesurée dans la colonne {column}.",
            },
        ]

        metrics: List[KPIMetric] = []
        for col in self.numeric_df.columns:
            series = self.numeric_df[col].dropna()
            if series.empty:
                continue
            col_lower = col.lower()
            for rule in rules:
                if any(keyword in col_lower for keyword in rule["keywords"]):
                    formatter = rule.get("formatter", _format_number)
                    value = rule["agg"](series)
                    formatted_value = formatter(value)
                    metrics.append(
                        KPIMetric(
                            label=f"{rule['label']} ({col})",
                            value=formatted_value,
                            column=col,
                            description=rule["description"].format(column=col),
                        )
                    )
                    break
            else:
                # No keyword match: propose generic statistics for prominent columns
                formatter = _format_number
                metrics.extend(
                    [
                        KPIMetric(
                            label=f"Moyenne ({col})",
                            value=formatter(series.mean()),
                            column=col,
                            description=f"Valeur moyenne observée pour {col}.",
                        ),
                        KPIMetric(
                            label=f"Somme ({col})",
                            value=formatter(series.sum()),
                            column=col,
                            description=f"Somme totale des valeurs de {col}.",
                        ),
                    ]
                )
        return metrics

    def _categorical_metrics(self) -> Iterable[KPIMetric]:
        if self.categorical_df.empty:
            return []

        metrics: List[KPIMetric] = []
        rules = [
            {
                "keywords": ["client", "customer", "user", "utilisateur", "compte"],
                "label": "Clients uniques",
                "description": "Nombre de clients/identifiants uniques présents dans {column}.",
            },
            {
                "keywords": ["produit", "product", "item", "article"],
                "label": "Produits uniques",
                "description": "Nombre de produits distincts présents dans {column}.",
            },
            {
                "keywords": ["ville", "city", "pays", "country", "region"],
                "label": "Localisations uniques",
                "description": "Nombre d'emplacements distincts détectés dans {column}.",
            },
        ]

        for col in self.categorical_df.columns:
            series = self.categorical_df[col].dropna()
            if series.empty:
                continue
            col_lower = col.lower()
            for rule in rules:
                if any(keyword in col_lower for keyword in rule["keywords"]):
                    metrics.append(
                        KPIMetric(
                            label=f"{rule['label']} ({col})",
                            value=_format_number(series.nunique(), decimals=0),
                            column=col,
                            description=rule["description"].format(column=col),
                        )
                    )
                    break
        return metrics

    def _datetime_metrics(self) -> Iterable[KPIMetric]:
        datetime_cols = self.df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns
        if not len(datetime_cols):
            return []

        metrics: List[KPIMetric] = []
        for col in datetime_cols:
            series = pd.to_datetime(self.df[col], errors="coerce").dropna()
            if series.empty:
                continue
            metrics.append(
                KPIMetric(
                    label=f"Date la plus récente ({col})",
                    value=series.max().strftime("%Y-%m-%d"),
                    column=col,
                    description=f"Dernière date disponible dans {col}.",
                )
            )
            metrics.append(
                KPIMetric(
                    label=f"Date la plus ancienne ({col})",
                    value=series.min().strftime("%Y-%m-%d"),
                    column=col,
                    description=f"Première date disponible dans {col}.",
                )
            )
        return metrics

    def _top_categories_details(self) -> Iterable[KPIDetail]:
        details: List[KPIDetail] = []
        if self.categorical_df.empty:
            return details

        for col in self.categorical_df.columns:
            series = self.categorical_df[col].dropna()
            unique_count = series.nunique()
            if unique_count == 0 or unique_count > 25:
                continue
            counts = series.value_counts().reset_index()
            counts.columns = [col, "Occurrences"]
            counts["Occurrences"] = counts["Occurrences"].astype(int)
            details.append(
                KPIDetail(
                    title=f"Répartition de {col}",
                    dataframe=counts,
                    description="Top catégories pour aider à identifier les segments dominants.",
                )
            )
        return details
