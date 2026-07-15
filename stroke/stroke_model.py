import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# 1) Veri setini oku
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

print("\n--- İLK 5 SATIR ---")
print(df.head())

print("\n--- INFO ---")
print(df.info())

print("\n--- NULL VALUES ---")
print(df.isnull().sum())

# 2) Eksik BMI değerlerini ortalama ile doldur
df["bmi"] = df["bmi"].fillna(df["bmi"].mean())

print("\n--- BMI TEMİZLENDİ ---")
print(df["bmi"].isnull().sum())

# 3) Kategorik verileri sayıya çevir
df = pd.get_dummies(df, drop_first=True)

print("\n--- ENCODED DATA ---")
print(df.head())

# 4) id sütununu çıkar
df = df.drop("id", axis=1)

# 5) Hedef ve özellikleri ayır
X = df.drop("stroke", axis=1)
y = df["stroke"]

# 6) Eğitim ve test verisini ayır
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7) Sadece eğitim verisine SMOTE uygula
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\n--- TRAIN CLASS DISTRIBUTION AFTER SMOTE ---")
print(y_train.value_counts())

# 8) Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 9) Modeller
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=3000,
        class_weight={0:1, 1:4},
        C=0.5,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=19
    )
}

best_model = None
best_name = ""
best_recall = 0
best_score = 0
best_threshold = 0.30
best_needs_scaler = False

for name, model in models.items():
    print(f"\n===== {name} =====")

    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        needs_scaler = True
    else:
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        needs_scaler = False

    # Stroke hastasını kaçırmamak için threshold biraz düşük
    y_pred = (y_proba > 0.20).astype(int)

    print("\n--- ACCURACY ---")
    print(accuracy_score(y_test, y_pred))

    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred))

    report = classification_report(y_test, y_pred, output_dict=True)
    recall_class_1 = report["1"]["recall"]

    precision_class_1 = report["1"]["precision"]
    f1_class_1 = report["1"]["f1-score"]

    score = (f1_class_1 * 0.7) + (recall_class_1 * 0.3)

    if score > best_score:
        best_score = score
        best_recall = recall_class_1
        best_model = model
        best_name = name
        best_needs_scaler = needs_scaler
print("\n=== EN İYİ MODEL ===")
print("En iyi model:", best_name)
print("Class 1 recall:", best_recall)
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

from sklearn.metrics import precision_score, recall_score, f1_score

thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

print("\n--- THRESHOLD ANALİZİ ---")

for threshold in thresholds:
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)

    precision = precision_score(y_test, y_pred_threshold, zero_division=0)
    recall = recall_score(y_test, y_pred_threshold, zero_division=0)
    f1 = f1_score(y_test, y_pred_threshold, zero_division=0)

    print(f"Threshold: {threshold}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")
    print("-" * 30)

with open("stroke_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

if best_needs_scaler:
    with open("stroke_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Scaler stroke_scaler.pkl olarak kaydedildi.")

print("En iyi model stroke_model.pkl olarak kaydedildi.")
print("\nX columns:")
print(list(X.columns))
print("Feature count:", len(X.columns))
print("EN İYİ MODEL:", best_model)