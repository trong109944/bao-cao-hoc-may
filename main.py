
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

# =====================================
# 1. TẠO DỮ LIỆU MẪU
# =====================================

np.random.seed(42)

def create_data(number):
    return pd.DataFrame({
        "students": np.random.randint(50, 2000, number),
        "rating": np.random.uniform(2.5, 5.0, number),
        "completion": np.random.uniform(40, 100, number),
        "difficulty": np.random.randint(1, 6, number),
        "duration": np.random.randint(5, 40, number)
    })

# Tạo 600 dữ liệu huấn luyện
train_data = create_data(600)

# Tạo 120 khóa học cần gợi ý
courses = create_data(120)

# =====================================
# 2. TẠO ĐIỂM ĐÁNH GIÁ MẪU
# =====================================

def calculate_score(data):
    score = (
        (data["students"] / 2000 * 100) * 0.30
        + ((data["rating"] - 1) / 4 * 100) * 0.25
        + data["completion"] * 0.25
        + ((6 - data["difficulty"]) / 5 * 100) * 0.10
        + ((40 - data["duration"]) / 35 * 100) * 0.10
    )

    return np.clip(
        score + np.random.normal(0, 3, len(data)),
        0, 100
    )

train_data["score"] = calculate_score(train_data)

# =====================================
# 3. CHUẨN BỊ DỮ LIỆU
# =====================================

features = [
    "students",
    "rating",
    "completion",
    "difficulty",
    "duration"
]

X_train = train_data[features]
y_train = train_data["score"]

X_test = courses[features]

# =====================================
# 4. LINEAR REGRESSION
# =====================================

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_result = linear_model.predict(X_test)

# =====================================
# 5. LOGISTIC REGRESSION
# =====================================

# Phân loại khóa học có điểm từ 60 trở lên
# là khóa học tiềm năng

y_class = (y_train >= 60).astype(int)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(max_iter=2000)

logistic_model.fit(X_train_scaled, y_class)

# Xác suất thuộc nhóm khóa học tiềm năng
logistic_result = (
    logistic_model.predict_proba(X_test_scaled)[:, 1] * 100
)

# =====================================
# 6. DECISION TREE
# =====================================

tree_model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

tree_model.fit(X_train, y_train)

tree_result = tree_model.predict(X_test)

# =====================================
# 7. TẠO BẢNG KẾT QUẢ
# =====================================

result = pd.DataFrame({
    "Khóa học": [
        f"CRS{i:03d}" for i in range(1, 121)
    ],
    "Linear Regression": linear_result,
    "Logistic Regression": logistic_result,
    "Decision Tree": tree_result
})

# =====================================
# 8. TÍNH ĐIỂM GỢI Ý
# =====================================

result["Điểm gợi ý"] = (
    result["Linear Regression"] * 0.4
    + result["Logistic Regression"] * 0.2
    + result["Decision Tree"] * 0.4
)

# Làm tròn 2 chữ số
score_columns = [
    "Linear Regression",
    "Logistic Regression",
    "Decision Tree",
    "Điểm gợi ý"
]

result[score_columns] = result[score_columns].round(2)

# Sắp xếp điểm gợi ý giảm dần
result = result.sort_values(
    by="Điểm gợi ý",
    ascending=False
).reset_index(drop=True)

# =====================================
# 9. HIỂN THỊ KẾT QUẢ
# =====================================

print("\n========== KẾT QUẢ GỢI Ý KHÓA HỌC ==========\n")

print(result.to_string(index=True))

# =====================================
# 10. XUẤT FILE CSV
# =====================================

result.to_csv(
    "ket_qua_goi_y.csv",
    index=False,
    encoding="utf-8-sig"
)

