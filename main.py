from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "course_recommendation_dataset1.csv"

FEATURES = [
    "gpa",
    "weekly_study_hours",
]
REQUIRED_COLUMNS = [
    "course_id",
    "predicted_score",
    "suitable",
    *FEATURES,
]


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {DATA_FILE}")

    data = pd.read_csv(DATA_FILE, sep=";", encoding="utf-8-sig")
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"CSV thiếu các cột bắt buộc: {', '.join(missing_columns)}")
    if data.empty:
        raise ValueError("CSV không có dữ liệu.")

    data["course_id"] = data["course_id"].astype("string").str.strip()
    if data["course_id"].isna().any() or data["course_id"].eq("").any():
        raise ValueError("CSV có dòng thiếu course_id.")

    numeric_columns = [*FEATURES, "predicted_score", "suitable"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")

    if data[numeric_columns].isna().any().any():
        raise ValueError("CSV có giá trị trống trong các cột dùng để huấn luyện.")

    classes = data["suitable"].unique()
    if len(classes) < 2:
        raise ValueError("Cột suitable cần có cả hai nhãn 0 và 1 để huấn luyện Logistic Regression.")

    return data


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    data = load_data()
    features = data[FEATURES]
    score_target = data["predicted_score"]
    suitable_target = data["suitable"].astype(int)

    logistic_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42),
    )
    cross_validation = KFold(n_splits=5, shuffle=True, random_state=42)
    linear_scores = cross_val_predict(
        LinearRegression(), features, score_target, cv=cross_validation
    )
    suitable_probabilities = (
        cross_val_predict(
            logistic_model,
            features,
            suitable_target,
            cv=cross_validation,
            method="predict_proba",
        )[:, 1]
        * 100
    )
    tree_scores = cross_val_predict(
        DecisionTreeRegressor(max_depth=5, random_state=42),
        features,
        score_target,
        cv=cross_validation,
    )

    score_max = max(float(score_target.max()), 1.0)
    predictions = pd.DataFrame(
        {
            "course_id": data["course_id"],
            "Linear Regression": np.clip(linear_scores / score_max * 100, 0, 100),
            "Logistic Regression": np.clip(suitable_probabilities, 0, 100),
            "Decision Tree": np.clip(tree_scores / score_max * 100, 0, 100),
        }
    )

    result = (
        predictions.groupby("course_id", as_index=False)[
            ["Linear Regression", "Logistic Regression", "Decision Tree"]
        ]
        .mean()
        .rename(columns={"course_id": "Khóa học"})
    )
    result["Điểm gợi ý"] = (
        result["Linear Regression"] * 0.4
        + result["Logistic Regression"] * 0.2
        + result["Decision Tree"] * 0.4
    )

    score_columns = [
        "Linear Regression",
        "Logistic Regression",
        "Decision Tree",
        "Điểm gợi ý",
    ]
    result[score_columns] = result[score_columns].round(2)
    result = result.sort_values("Điểm gợi ý", ascending=False).reset_index(drop=True)

    print("\n========== KẾT QUẢ GỢI Ý KHÓA HỌC ==========\n")
    print(result.to_string(index=True))


if __name__ == "__main__":
    main()
