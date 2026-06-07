#!/usr/bin/env python
"""Point d'entrée : figures, modèle, métriques et rapport IMRAD."""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.config import DATA_PATH, FIGURES_DIR, MODELS_DIR, RANDOM_STATE  # noqa: E402
from analysis.data import get_feature_columns, prepare_dataset  # noqa: E402
from analysis.modeling import (  # noqa: E402
    build_pipeline,
    build_preprocessor,
    compute_lrt,
    compute_odds_ratios,
    evaluate_baselines,
    extract_coefficients,
    log_likelihood_multinomial,
    run_grid_search,
    split_data,
)
from analysis.plots import (  # noqa: E402
    plot_class_distribution,
    plot_coefficients,
    plot_cv_results,
    plot_eda_figures,
    plot_evaluation,
    plot_odds_ratios,
)
from analysis.report import generate_imrad_report  # noqa: E402
from analysis.report_context import build_report_context, save_report_context  # noqa: E402


def main() -> dict:
    """Exécute la chaîne complète d'analyse."""
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette(["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df, n_ties = prepare_dataset(DATA_PATH)
    print(f"Dataset prepared: n={len(df)}, ties broken randomly: {n_ties}")

    plot_class_distribution(df)
    plot_eda_figures(df)

    ct = pd.crosstab(df["gender"], df["behavior_profile"])
    chi2, p_chi, dof, _ = chi2_contingency(ct)
    print(f"Chi2 gender x profile: stat={chi2:.3f}, p={p_chi:.4g}, dof={dof}")

    X_train, X_test, y_train, y_test = split_data(df)
    numeric_cols, categorical_cols = get_feature_columns()
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    pipeline = build_pipeline(preprocessor, C=1.0)

    grid, cv_results, best_c = run_grid_search(pipeline, X_train, y_train)
    plot_cv_results(cv_results, best_c)

    pipeline_final = build_pipeline(preprocessor, C=best_c)
    pipeline_final.fit(X_train, y_train)

    y_pred = pipeline_final.predict(X_test)
    y_proba = pipeline_final.predict_proba(X_test)
    classes = list(pipeline_final.named_steps["clf"].classes_)

    eval_metrics = plot_evaluation(y_test, y_pred, y_proba, classes)
    coef_df = extract_coefficients(pipeline_final)
    plot_coefficients(coef_df)
    or_df = compute_odds_ratios(coef_df)
    plot_odds_ratios(or_df)

    baselines = evaluate_baselines(y_train, y_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")

    # LRT : modèle complet vs modèle nul (probabilités a priori) sur le jeu d'entraînement
    prep = pipeline_final.named_steps["prep"]
    X_train_t = prep.fit_transform(X_train)
    X_train_t_const = np.hstack([np.ones((X_train_t.shape[0], 1)), X_train_t])
    proba_full = pipeline_final.predict_proba(X_train)

    n_classes = len(classes)
    n_features = X_train_t.shape[1]
    df_diff = (n_classes - 1) * n_features

    log_lik_full = log_likelihood_multinomial(y_train, proba_full, classes)
    prior = y_train.value_counts(normalize=True).reindex(classes).to_numpy()
    proba_null = np.tile(prior, (len(y_train), 1))
    log_lik_null = log_likelihood_multinomial(y_train, proba_null, classes)
    lrt = compute_lrt(log_lik_full, log_lik_null, df_diff)

    model_path = MODELS_DIR / "multinomial_lr_model.pkl"
    joblib.dump(pipeline_final, model_path)

    summary = {
        "n_obs": len(df),
        "n_ties": n_ties,
        "best_c": best_c,
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "baseline_accuracy": float(baselines["baseline_accuracy"]),
        "baseline_f1_macro": float(baselines["baseline_f1_macro"]),
        "lrt_statistic": lrt["lr_statistic"],
        "lrt_p": lrt["p_value"],
        "chi2_gender_p": float(p_chi),
        "class_counts": df["behavior_profile"].value_counts().to_dict(),
        "cv_results": cv_results.to_dict(orient="records"),
    }

    results_path = PROJECT_ROOT / "outputs" / "results_summary.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    ctx = build_report_context(
        df=df,
        n_ties=n_ties,
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
        y_test_labels=y_test,
        classes=classes,
        pipeline_final=pipeline_final,
        coef_df=coef_df,
        cv_results=cv_results,
        best_c=best_c,
        accuracy=accuracy,
        f1_macro=f1_macro,
        baselines=baselines,
        lrt=lrt,
        chi2_stat=chi2,
        p_chi=p_chi,
        dof=dof,
        n_raw=481,
        n_features_encoded=X_train_t.shape[1],
    )
    save_report_context(ctx)
    generate_imrad_report(ctx)

    print(f"Model saved: {model_path}")
    print(f"Accuracy: {accuracy:.4f}, F1 macro: {f1_macro:.4f}")
    print(f"LRT p-value: {lrt['p_value']:.4g}")
    return summary


if __name__ == "__main__":
    main()
