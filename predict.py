import pickle
import numpy as np

# Modeli yükle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

print("🔮 Diyabet Tahmin Sistemi")

# Kullanıcıdan veri al
pregnancies = float(input("Gebelik sayısı: "))
glucose = float(input("Glucose: "))
blood_pressure = float(input("Blood Pressure: "))
skin_thickness = float(input("Skin Thickness: "))
insulin = float(input("Insulin: "))
bmi = float(input("BMI: "))
dpf = float(input("Diabetes Pedigree Function: "))
age = float(input("Yaş: "))

# Veriyi modele uygun hale getir
data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                  insulin, bmi, dpf, age]])

# Tahmin yap
prediction = model.predict(data)
probability = model.predict_proba([data])[0][1]
# Sonuç
if prediction[0] == 1:
    result = "Diyabet riski VAR"
else:
    result = "Diyabet riski YOK"

print(result)
print("Risk oranı:", probability * 100)