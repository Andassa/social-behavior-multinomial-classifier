"""Chemins, hyperparamètres et constantes du projet."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "smmh.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
REPORT_DIR = PROJECT_ROOT / "report"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
DPI = 300

PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
PROFILE_ORDER = [
    "ADHD-Dominant",
    "Anxiety-Dominant",
    "LowSelfEsteem-Dominant",
    "Depression-Dominant",
]
PROFILE_LABELS_FR = {
    "ADHD-Dominant": "Profil ADHD dominant",
    "Anxiety-Dominant": "Profil Anxiété dominant",
    "LowSelfEsteem-Dominant": "Profil faible estime de soi",
    "Depression-Dominant": "Profil Dépression dominant",
}

TIME_MAP = {
    "Less than an Hour": 1,
    "Between 1 and 2 hours": 2,
    "Between 2 and 3 hours": 3,
    "Between 3 and 4 hours": 4,
    "Between 4 and 5 hours": 5,
    "More than 5 hours": 6,
}

TOP_PLATFORMS = [
    "Facebook",
    "Instagram",
    "YouTube",
    "Twitter",
    "TikTok",
    "Snapchat",
    "Discord",
    "Reddit",
    "Pinterest",
]

AUTHOR_NAME = "RAMANANANDRO Andassa Fanomezantsoa"
AUTHOR_ID = "MII 227I22"
INSTITUTION = "EMIT Fianarantsoa"

COLUMN_RENAME = {
    "Timestamp": "timestamp",
    "1. What is your age?": "age",
    "2. Gender": "gender",
    "3. Relationship Status": "relationship",
    "4. Occupation Status": "occupation",
    "5. What type of organizations are you affiliated with?": "organization",
    "6. Do you use social media?": "uses_social_media",
    "7. What social media platforms do you commonly use?": "platforms",
    "8. What is the average time you spend on social media every day?": "time_on_social_media",
    "9. How often do you find yourself using Social media without a specific purpose?": "q9_no_purpose",
    "10. How often do you get distracted by Social media when you are busy doing something?": "q10_distracted",
    "11. Do you feel restless if you haven't used Social media in a while?": "q11_restless",
    "12. On a scale of 1 to 5, how easily distracted are you?": "q12_easily_distracted",
    "13. On a scale of 1 to 5, how much are you bothered by worries?": "q13_worries",
    "14. Do you find it difficult to concentrate on things?": "q14_concentrate",
    "15. On a scale of 1-5, how often do you compare yourself to other successful people through the use of social media?": "q15_compare",
    "16. Following the previous question, how do you feel about these comparisons, generally speaking?": "q16_comparison_feeling",
    "17. How often do you look to seek validation from features of social media?": "q17_validation",
    "18. How often do you feel depressed or down?": "q18_depressed",
    "19. On a scale of 1 to 5, how frequently does your interest in daily activities fluctuate?": "q19_interest",
    "20. On a scale of 1 to 5, how often do you face issues regarding sleep?": "q20_sleep",
}

LIKERT_COLS = [
    "q9_no_purpose",
    "q10_distracted",
    "q11_restless",
    "q12_easily_distracted",
    "q13_worries",
    "q14_concentrate",
    "q15_compare",
    "q16_comparison_feeling",
    "q17_validation",
    "q18_depressed",
    "q19_interest",
    "q20_sleep",
]

SCORE_COLS = [
    "score_ADHD",
    "score_anxiety",
    "score_self_esteem",
    "score_depression",
]
