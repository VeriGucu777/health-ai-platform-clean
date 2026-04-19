import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Veri yükle
df = pd.read_csv("diabetes.csv")

# Temizleme
columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in columns:
    df[col] = df[col].replace(0, df[col].mean())

# Feature - Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train - Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model oluştur
model = LogisticRegression()
model.fit(X_train, y_train)

# 🔥 MODELİ KAYDET
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model başarıyla kaydedildi!")
