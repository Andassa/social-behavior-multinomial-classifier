"""EDA and evaluation figure generation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from analysis.config import (
    DPI,
    FIGURES_DIR,
    PALETTE,
    PROFILE_ORDER,
    SCORE_COLS,
    TOP_PLATFORMS,
)


def save_fig(name: str, dpi: int = DPI) -> Path:
    """Save current matplotlib figure to outputs/figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path


def plot_class_distribution(df: pd.DataFrame) -> None:
    """Pie and bar charts of behavior profiles."""
    counts = df["behavior_profile"].value_counts().reindex(PROFILE_ORDER)
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=PALETTE, startangle=90)
    plt.title("Distribution des profils comportementaux")
    save_fig("fig03b_class_distribution_pie")

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar", color=PALETTE)
    plt.title("Effectifs par profil comportemental")
    plt.ylabel("Effectifs")
    plt.xticks(rotation=25)
    save_fig("fig03c_class_distribution_bar")


def plot_eda_figures(df: pd.DataFrame) -> None:
    """Generate fig01-fig08 exploratory plots."""
    numeric_cols = ["age", "time_social_media_ord", "n_platforms"] + SCORE_COLS
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Matrice de corrélation — variables numériques et scores")
    save_fig("fig01_correlation_heatmap")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, score, color in zip(
        axes.ravel(), SCORE_COLS, PALETTE
    ):
        sns.violinplot(
            data=df, x="behavior_profile", y=score, order=PROFILE_ORDER,
            hue="behavior_profile", palette=PALETTE, legend=False, ax=ax, inner="quartile"
        )
        ax.set_title(score)
        ax.tick_params(axis="x", rotation=20)
    plt.suptitle("Distribution des scores composites par profil")
    save_fig("fig02_violin_scores_by_profile")

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df, x="behavior_profile", y="time_social_media_ord",
        order=PROFILE_ORDER, hue="behavior_profile", palette=PALETTE, legend=False
    )
    plt.title("Temps quotidien sur les réseaux vs profil")
    plt.xlabel("Profil comportemental")
    plt.ylabel("Temps (échelle ordinale)")
    save_fig("fig03_boxplot_time_vs_profile")

    g = sns.pairplot(
        df, vars=SCORE_COLS, hue="behavior_profile",
        hue_order=PROFILE_ORDER, palette=PALETTE, corner=True, plot_kws={"alpha": 0.6}
    )
    g.fig.suptitle("Pairplot des scores composites", y=1.02)
    g.savefig(FIGURES_DIR / "fig04_pairplot_scores.png", dpi=DPI, bbox_inches="tight")
    plt.close("all")

    platform_rows = []
    for profile in PROFILE_ORDER:
        subset = df[df["behavior_profile"] == profile]
        for platform in TOP_PLATFORMS:
            platform_rows.append({
                "profile": profile,
                "platform": platform,
                "count": subset["platforms"].fillna("").str.contains(platform, case=False).sum(),
            })
    plat_df = pd.DataFrame(platform_rows)
    top = plat_df.groupby("platform")["count"].sum().sort_values(ascending=False).head(6).index
    plat_df = plat_df[plat_df["platform"].isin(top)]
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plat_df, x="platform", y="count", hue="profile", hue_order=PROFILE_ORDER, palette=PALETTE)
    plt.title("Plateformes les plus utilisées par profil")
    plt.ylabel("Nombre d'utilisateurs")
    save_fig("fig05_platforms_by_profile")

    plt.figure(figsize=(10, 6))
    for profile, color in zip(PROFILE_ORDER, PALETTE):
        subset = df[df["behavior_profile"] == profile]["age"]
        plt.hist(subset, bins=15, alpha=0.5, label=profile, color=color)
    plt.title("Distribution de l'âge par profil")
    plt.xlabel("Âge")
    plt.ylabel("Fréquence")
    plt.legend()
    save_fig("fig06_age_by_profile")

    ct = pd.crosstab(df["gender"], df["behavior_profile"]).reindex(columns=PROFILE_ORDER, fill_value=0)
    ct.plot(kind="bar", stacked=True, color=PALETTE, figsize=(10, 6))
    plt.title("Genre × profil comportemental")
    plt.ylabel("Effectifs")
    plt.xticks(rotation=0)
    save_fig("fig07_gender_stacked_profile")

    rel_ct = pd.crosstab(df["relationship"], df["behavior_profile"])
    rel_ct = rel_ct.reindex(columns=PROFILE_ORDER, fill_value=0)
    plt.figure(figsize=(12, 7))
    sns.heatmap(rel_ct, annot=True, fmt="d", cmap="Blues")
    plt.title("Statut relationnel × profil comportemental")
    save_fig("fig08_relationship_heatmap")


def plot_cv_results(cv_results: pd.DataFrame, best_c: float) -> None:
    """Plot accuracy vs log(C)."""
    plt.figure(figsize=(8, 5))
    plt.semilogx(cv_results["C"], cv_results["mean_accuracy"], marker="o", label="Moyenne CV")
    plt.axvline(best_c, color="red", linestyle="--", label=f"C optimal = {best_c}")
    plt.xlabel("C (échelle log)")
    plt.ylabel("Accuracy")
    plt.title("Validation croisée : accuracy vs C")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_fig("fig09_cv_accuracy_vs_C")


def plot_evaluation(
    y_test, y_pred, y_proba, classes, feature_names=None
) -> dict:
    """Generate evaluation figures and return metrics dict."""
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Purples",
        xticklabels=classes, yticklabels=classes
    )
    plt.title("Matrice de confusion normalisée")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    save_fig("fig10_confusion_matrix")

    report = classification_report(y_test, y_pred, labels=classes, output_dict=True)
    metrics_df = pd.DataFrame(report).T.loc[classes, ["precision", "recall", "f1-score"]]
    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, metrics_df["precision"], width, label="Precision", color=PALETTE[0])
    ax.bar(x, metrics_df["recall"], width, label="Recall", color=PALETTE[1])
    ax.bar(x + width, metrics_df["f1-score"], width, label="F1", color=PALETTE[2])
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_title("Métriques de classification par profil")
    ax.set_ylabel("Score")
    ax.legend()
    save_fig("fig11_classification_metrics")

    y_bin = label_binarize(y_test, classes=classes)
    plt.figure(figsize=(8, 6))
    for i, (cls, color) in enumerate(zip(classes, PALETTE)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, label=f"{cls} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Courbes ROC — One-vs-Rest")
    plt.legend(loc="lower right")
    save_fig("fig12_roc_ovr")

    plt.figure(figsize=(8, 6))
    for i, (cls, color) in enumerate(zip(classes, PALETTE)):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        plt.plot(rec, prec, color=color, label=cls)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Courbes Precision–Recall par classe")
    plt.legend()
    save_fig("fig13_precision_recall")

    return {
        "classification_report": report,
        "metrics_df": metrics_df,
        "confusion_matrix": cm,
    }


def plot_coefficients(coef_df: pd.DataFrame) -> None:
    """Heatmap of multinomial coefficients."""
    plt.figure(figsize=(14, max(6, len(coef_df) * 0.25)))
    sns.heatmap(coef_df, cmap="RdBu_r", center=0, annot=False)
    plt.title("Heatmap des coefficients β (features × classes)")
    plt.xlabel("Classe")
    plt.ylabel("Feature")
    save_fig("fig14_coefficients_heatmap")


def plot_odds_ratios(or_df: pd.DataFrame, top_n: int = 8) -> None:
    """Forest-style plot of top odds ratios per class."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, cls, color in zip(axes.ravel(), or_df.columns, PALETTE):
        sub = or_df[cls].dropna()
        sub = sub.iloc[np.argsort(np.abs(sub))[-top_n:]]
        sub.plot(kind="barh", ax=ax, color=color)
        ax.set_title(f"Top OR — {cls}")
        ax.set_xlabel("exp(β)")
    plt.suptitle("Odds ratios — features les plus influentes")
    save_fig("fig15_odds_ratios")
