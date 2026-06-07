# Social Behavior Multinomial Classifier

## Description

Classification multinomiale des profils comportementaux liés aux réseaux sociaux via **régression logistique multinomiale** (fonction softmax). Projet académique avec notebook commenté, figures d'analyse (300 dpi) et export des métriques.

**Auteur :** RAMANANANDRO Andassa Fanomezantsoa (MII 227I22) — EMIT Fianarantsoa

## Dataset

[Social Media and Mental Health](https://www.kaggle.com/datasets/souvikahmed071/social-media-and-mental-health)

- **481** réponses brutes · **478** après filtrage (`uses_social_media == Yes`)
- **21** variables questionnaire · **4** profils comportementaux cible

Fichier local : `data/smmh.csv`

## Méthode

1. **Scores composites** (Likert 1–5) :
   - ADHD : Q9 + Q10 + Q12
   - Anxiété : Q11 + Q13
   - Estime de soi : Q15 + Q16 + Q17
   - Dépression : Q18 + Q19 + Q20
2. **Cible** `behavior_profile` = argmax des 4 scores ; **égalités** résolues par `np.random.default_rng(42)` (100 cas sur 478).
3. **Features X** : démographie + usage SM (temps, plateformes) — **sans** les Likert utilisés pour la cible (anti-fuite).
4. **Pipeline** : `StandardScaler` + `OneHotEncoder` → `LogisticRegression(solver='lbfgs')`.
5. **Validation** : `StratifiedKFold` k=5, `GridSearchCV` sur `C ∈ {0.001, …, 100}`.
6. **Simulation** interactive `ipywidgets` dans le notebook.

## Structure

```
social-behavior-multinomial-classifier/
├── README.md
├── requirements.txt
├── run_pipeline.py              # Exécution bout-en-bout (figures + modèle + métriques)
├── analysis/                    # Modules réutilisables
├── data/smmh.csv
├── notebooks/social_behavior_classifier.ipynb
├── outputs/figures/             # fig01 … fig15 (PNG 300 dpi)
├── outputs/results_summary.json
└── outputs/report_context.json
```

## Installation et exécution

```bash
cd social-behavior-multinomial-classifier
pip install -r requirements.txt
python run_pipeline.py
```

Notebook interactif :

```bash
jupyter notebook notebooks/social_behavior_classifier.ipynb
```

## Résultats clés (exécution reproductible, `random_state=42`)

| Métrique | Valeur |
|----------|--------|
| Observations | 478 |
| Égalités argmax (rng=42) | 100 |
| C optimal (GridSearchCV) | 0.001 |
| Accuracy test | 0.396 |
| F1 macro | 0.142 |
| Baseline ZeroR accuracy | 0.396 |
| Chi² genre × profil (p) | 0.333 |
| LRT statistique | 10.43 |

**Distribution des classes :** ADHD-Dominant 187 · Depression-Dominant 138 · Anxiety-Dominant 92 · LowSelfEsteem-Dominant 61

> La performance proche du baseline reflète la difficulté de prédire un profil construit à partir de symptômes internes en n'utilisant que des variables exogènes (âge, genre, temps SM, plateformes) — choix méthodologique volontaire pour éviter la fuite de données.

## Théories mathématiques couvertes

- Fonction **Softmax**
- **Log-vraisemblance** multinomiale
- Algorithme **L-BFGS**
- Régularisation **L2** (Ridge, paramètre `C`)
- **Odds ratios** multinomiaux
- Test du rapport de vraisemblance (**LRT**)
- Métriques multiclasses (**F1**, matrice de confusion, **AUC-ROC** OvR)

## Licence

Projet académique — dataset Kaggle soumis aux conditions de l’auteur original.
