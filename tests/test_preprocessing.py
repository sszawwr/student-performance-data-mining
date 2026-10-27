import pandas as pd
from src.preprocessing import clean_student_data, select_features

def test_clean_removes_duplicates_and_missing_target():
    df = pd.DataFrame({"studytime": [2, 2, 3], "G3": [10, 10, None]})
    clean = clean_student_data(df)
    assert len(clean) == 1
    assert clean["G3"].iloc[0] == 10

def test_target_not_in_features():
    df = pd.DataFrame({"studytime": [1, 2], "G1": [5, 10], "G2": [6, 11], "G3": [7, 12]})
    X, y = select_features(df, include_previous_grades=False)
    assert "G3" not in X.columns
    assert "G1" not in X.columns and "G2" not in X.columns
    assert len(y) == 2
