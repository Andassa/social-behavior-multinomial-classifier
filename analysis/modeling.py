"""Entraînement, validation croisée et inférence du classifieur multinomial."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analysis.config import CV_FOLDS, MODELS_DIR, RANDOM_STATE, TEST_SIZE
from analysis.data import get_feature_columns


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Préprocesseur : standardisation numérique + encodage one-hot."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def build_pipeline(preprocessor: ColumnTransformer, C: float = 1.0) -> Pipeline:
    """Pipeline sklearn : prétraitement puis régression logistique multinomiale."""
    return Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                    C=C,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Noms des variables après transformation (numériques + dummies)."""
    num_features = preprocessor.transformers_[0][2]
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_features = list(cat_encoder.get_feature_names_out())
    return list(num_features) + cat_features


def run_grid_search(pipeline: Pipeline, X_train, y_train) -> tuple[GridSearchCV, pd.DataFrame, float]:
    """Recherche par grille du paramètre de régularisation C (validation croisée stratifiée)."""
    param_grid = {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100]}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        pipeline, param_grid=param_grid, cv=cv, scoring="accuracy", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    results = pd.DataFrame(grid.cv_results_)[["param_clf__C", "mean_test_score"]]
    results = results.rename(columns={"param_clf__C": "C", "mean_test_score": "mean_accuracy"})
    best_c = float(grid.best_params_["clf__C"])
    return grid, results, best_c


def compute_lrt(log_lik_full: float, log_lik_null: float, df_diff: int) -> dict:
    """Statistique du test du rapport de vraisemblance (LR, ddl, p-value)."""
    lr_stat = -2 * (log_lik_null - log_lik_full)
    p_value = float(1 - stats.chi2.cdf(lr_stat, df_diff))
    return {"lr_statistic": lr_stat, "df": df_diff, "p_value": p_value}


def log_likelihood_multinomial(y_true, proba, classes) -> float:
    """Log-vraisemblance multinomiale sur les probabilités prédites."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx = np.array([class_to_idx[y] for y in y_true])
    proba_clipped = np.clip(proba, 1e-15, 1.0)
    return float(np.sum(np.log(proba_clipped[np.arange(len(y_true)), idx])))


def evaluate_baselines(y_train, y_test) -> dict:
    """Référence ZeroR (classe majoritaire) : accuracy et F1 macro."""
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    y_pred_dummy = dummy.predict(np.zeros((len(y_test), 1)))
    return {
        "baseline_accuracy": accuracy_score(y_test, y_pred_dummy),
        "baseline_f1_macro": f1_score(y_test, y_pred_dummy, average="macro"),
    }


def extract_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    """Tableau des coefficients β (features × classes)."""
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    feature_names = get_feature_names(prep)
    coef = clf.coef_
    classes = clf.classes_
    return pd.DataFrame(coef.T, index=feature_names, columns=classes)


def compute_odds_ratios(
    coef_df: pd.DataFrame,
    reference_class: str | None = None,
) -> pd.DataFrame:
    """Odds ratios par paire : exp(β_k − β_ref) par rapport à la classe de référence."""
    ref = reference_class or coef_df.columns[-1]
    ref_coef = coef_df[ref]
    return pd.DataFrame(
        {cls: np.exp(coef_df[cls] - ref_coef) for cls in coef_df.columns if cls != ref},
        index=coef_df.index,
    )


def split_data(df: pd.DataFrame, target_col: str = "behavior_profile"):
    """Partition train/test stratifiée sur la variable cible."""
    numeric_cols, categorical_cols = get_feature_columns()
    feature_cols = numeric_cols + categorical_cols
    X = df[feature_cols]
    y = df[target_col]
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
