"""Assemblage du contexte numérique pour le rapport Word IMRAD."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import classification_report

from analysis.config import (
    COLUMN_RENAME,
    PROFILE_ORDER,
    PROJECT_ROOT,
    RANDOM_STATE,
)


@dataclass
class ReportContext:
    """Données chiffrées et métadonnées nécessaires à la rédaction de l'exposé."""

    n_obs: int
    n_raw: int
    n_ties: int
    best_c: float
    accuracy: float
    f1_macro: float
    baseline_accuracy: float
    baseline_f1_macro: float
    lrt_statistic: float
    lrt_p: float
    chi2_gender: float
    chi2_gender_p: float
    chi2_gender_dof: int
    age_mean: float
    age_min: float
    age_max: float
    age_std: float
    gender_counts: dict[str, int]
    class_counts: dict[str, int]
    class_pct: dict[str, float]
    classification_rows: list[dict[str, Any]]
    cv_results: list[dict[str, Any]]
    top_coef_by_class: dict[str, list[tuple[str, float]]]
    train_size: int
    test_size: int
    n_features_encoded: int
    figures_body: list[tuple[str, str]] = field(default_factory=list)
    figures_annex: list[tuple[str, str]] = field(default_factory=list)


def _profile_percentages(counts: dict[str, int], total: int) -> dict[str, float]:
    return {p: round(100 * counts.get(p, 0) / total, 1) for p in PROFILE_ORDER}


def build_report_context(
    df: pd.DataFrame,
    n_ties: int,
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred,
    y_test_labels,
    classes: list,
    pipeline_final,
    coef_df: pd.DataFrame,
    cv_results: pd.DataFrame,
    best_c: float,
    accuracy: float,
    f1_macro: float,
    baselines: dict,
    lrt: dict,
    chi2_stat: float,
    p_chi: float,
    dof: int,
    n_raw: int = 481,
    n_features_encoded: int = 0,
) -> ReportContext:
    """Consolide les sorties du pipeline dans une structure ReportContext."""
    report_dict = classification_report(
        y_test_labels, y_pred, labels=classes, output_dict=True, zero_division=0
    )
    classification_rows = []
    for cls in classes:
        row = report_dict.get(cls, {})
        if isinstance(row, dict):
            classification_rows.append({
                "class": cls,
                "precision": round(row.get("precision", 0), 3),
                "recall": round(row.get("recall", 0), 3),
                "f1": round(row.get("f1-score", 0), 3),
                "support": int(row.get("support", 0)),
            })

    counts = df["behavior_profile"].value_counts().to_dict()
    total = len(df)

    top_coef_by_class = {}
    for cls in coef_df.columns:
        s = coef_df[cls].abs().sort_values(ascending=False).head(5)
        top_coef_by_class[cls] = [(idx, round(float(coef_df.loc[idx, cls]), 4)) for idx in s.index]

    figures_body = [
        ("fig03c_class_distribution_bar", "Distribution des profils comportementaux (effectifs)"),
        ("fig10_confusion_matrix", "Matrice de confusion normalisée sur l'échantillon test"),
        ("fig11_classification_metrics", "Précision, rappel et F1-score par profil"),
        ("fig12_roc_ovr", "Courbes ROC en approche One-vs-Rest"),
        ("fig09_cv_accuracy_vs_C", "Accuracy moyenne en validation croisée selon le paramètre C"),
        ("fig14_coefficients_heatmap", "Carte des coefficients β (features × classes)"),
    ]
    figures_annex = [
        ("fig03b_class_distribution_pie", "Répartition en secteurs des quatre profils"),
        ("fig01_correlation_heatmap", "Matrice de corrélation des variables numériques"),
        ("fig02_violin_scores_by_profile", "Scores composites par profil (violons)"),
        ("fig03_boxplot_time_vs_profile", "Temps sur les réseaux sociaux selon le profil"),
        ("fig04_pairplot_scores", "Pairplot des quatre scores composites"),
        ("fig05_platforms_by_profile", "Plateformes les plus citées par profil"),
        ("fig06_age_by_profile", "Distribution de l'âge par profil"),
        ("fig07_gender_stacked_profile", "Répartition genre × profil"),
        ("fig08_relationship_heatmap", "Statut relationnel × profil comportemental"),
        ("fig13_precision_recall", "Courbes précision–rappel par classe"),
        ("fig15_odds_ratios", "Odds ratios des principales variables explicatives"),
    ]

    return ReportContext(
        n_obs=total,
        n_raw=n_raw,
        n_ties=n_ties,
        best_c=float(best_c),
        accuracy=float(accuracy),
        f1_macro=float(f1_macro),
        baseline_accuracy=float(baselines["baseline_accuracy"]),
        baseline_f1_macro=float(baselines["baseline_f1_macro"]),
        lrt_statistic=float(lrt["lr_statistic"]),
        lrt_p=float(lrt["p_value"]),
        chi2_gender=float(chi2_stat),
        chi2_gender_p=float(p_chi),
        chi2_gender_dof=int(dof),
        age_mean=float(df["age"].mean()),
        age_min=float(df["age"].min()),
        age_max=float(df["age"].max()),
        age_std=float(df["age"].std()),
        gender_counts=df["gender"].value_counts().to_dict(),
        class_counts=counts,
        class_pct=_profile_percentages(counts, total),
        classification_rows=classification_rows,
        cv_results=cv_results.to_dict(orient="records"),
        top_coef_by_class=top_coef_by_class,
        train_size=len(y_train),
        test_size=len(y_test),
        n_features_encoded=n_features_encoded,
        figures_body=figures_body,
        figures_annex=figures_annex,
    )


def save_report_context(ctx: ReportContext, path: Path | None = None) -> Path:
    """Écrit le contexte rapport au format JSON (outputs/report_context.json)."""
    path = path or PROJECT_ROOT / "outputs" / "report_context.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(ctx), f, indent=2, ensure_ascii=False)
    return path


def load_report_context(path: Path | None = None) -> ReportContext:
    """Charge le contexte rapport depuis le fichier JSON."""
    path = path or PROJECT_ROOT / "outputs" / "report_context.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return ReportContext(**data)


def get_variable_annex_rows() -> list[tuple[str, str, str]]:
    """Inventaire des variables du questionnaire pour l'annexe A."""
    descriptions = {
        "timestamp": "Horodatage de la réponse",
        "age": "Âge en années",
        "gender": "Genre déclaré",
        "relationship": "Statut relationnel",
        "occupation": "Statut professionnel",
        "organization": "Type d'organisation affiliée",
        "uses_social_media": "Utilisation des réseaux sociaux (Oui/Non)",
        "platforms": "Plateformes utilisées (texte multi-valeurs)",
        "time_on_social_media": "Temps quotidien sur les réseaux",
        "q9_no_purpose": "Usage sans but précis (Likert 1–5)",
        "q10_distracted": "Distraction par les réseaux (Likert 1–5)",
        "q11_restless": "Agitation en l'absence de réseaux (Likert 1–5)",
        "q12_easily_distracted": "Facilité de distraction (Likert 1–5)",
        "q13_worries": "Préoccupations / inquiétude (Likert 1–5)",
        "q14_concentrate": "Difficulté de concentration",
        "q15_compare": "Comparaison sociale (Likert 1–5)",
        "q16_comparison_feeling": "Sentiment après comparaison (Likert 1–5)",
        "q17_validation": "Recherche de validation (Likert 1–5)",
        "q18_depressed": "Humeur dépressive (Likert 1–5)",
        "q19_interest": "Fluctuation de l'intérêt (Likert 1–5)",
        "q20_sleep": "Problèmes de sommeil (Likert 1–5)",
    }
    rows = []
    for orig, short in COLUMN_RENAME.items():
        dtype = "Catégoriel" if short in {
            "gender", "relationship", "occupation", "organization",
            "uses_social_media", "platforms", "time_on_social_media",
        } else "Numérique / ordinal"
        rows.append((short, dtype, descriptions.get(short, orig)))
    return rows
