"""Preprocessing and feature selection."""
import pandas as pd

TARGET = "G3"
GRADE_FEATURES = ["G1", "G2"]

def clean_student_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column labels, remove duplicate rows, and drop rows without target."""
    clean = df.copy()
    clean.columns = [str(col).strip() for col in clean.columns]
    clean = clean.drop_duplicates()
    if TARGET not in clean.columns:
        raise ValueError(f"Kolom target {TARGET} tidak ditemukan.")
    clean[TARGET] = pd.to_numeric(clean[TARGET], errors="coerce")
    clean = clean.dropna(subset=[TARGET])
    return clean.reset_index(drop=True)

def select_features(df: pd.DataFrame, include_previous_grades: bool = False):
    """Return X/y. Excluding G1/G2 tests prediction without earlier period grades."""
    excluded = [TARGET]
    if not include_previous_grades:
        excluded += GRADE_FEATURES
    X = df.drop(columns=excluded, errors="ignore")
    y = df[TARGET].astype(float)
    return X, y
