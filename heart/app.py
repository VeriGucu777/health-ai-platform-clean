from flask import Flask, render_template, request
import pickle
import numpy as np
from grafik import grafik_ciz
from pdf_olustur import pdf_olustur
app = Flask(__name__)

# Modeli yükle
model = pickle.load(open("model.pkl", "rb"))

# Ana sayfa
@app.route("/")
def home():
    return render_template("heart_index.html")


# Temizleme fonksiyonu
def temizle(x):
    try:
        return float(x)
    except:
        return 0


# Tahmin route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        yas = temizle(request.form.get("yas"))
        cinsiyet = temizle(request.form.get("cinsiyet"))
        gogus_agrisi = temizle(request.form.get("gogus_agrisi"))
        tansiyon = temizle(request.form.get("tansiyon"))
        kolesterol = temizle(request.form.get("kolesterol"))
        kan_sekeri = temizle(request.form.get("kan_sekeri"))
        ekg = temizle(request.form.get("ekg"))
        max_nabiz = temizle(request.form.get("max_nabiz"))
        egzersiz_anlina = temizle(request.form.get("egzersiz_anlina"))
        st_depresyon = temizle(request.form.get("st_depresyon"))
        egim = temizle(request.form.get("egim"))
        damar_sayisi = temizle(request.form.get("damar_sayisi"))
        thal = temizle(request.form.get("thal"))

        veri = np.array([[yas, cinsiyet, gogus_agrisi, tansiyon, kolesterol,
                          kan_sekeri, ekg, max_nabiz, egzersiz_anlina,
                          st_depresyon, egim, damar_sayisi, thal]])

        feature_isimleri = [
            "Yaş", "Cinsiyet", "Göğüs Ağrısı", "Tansiyon",
            "Kolesterol", "Kan Şekeri", "EKG", "Max Nabız",
            "Egzersiz", "ST Depresyon", "Eğim", "Damar", "Thal"
        ]

        grafik_ciz(veri, feature_isimleri)

        print("VERİ:", veri)

        olasilik = model.predict_proba(veri)[0]
        print("OLASILIK:", olasilik)

        # eşik
        risk_orani = round(olasilik[1] * 100, 2)
        if olasilik[1] > 0.70:
            sonuc = f"🚨 YÜKSEK RİSK (%{risk_orani})"
        elif olasilik[1] > 0.30:
            sonuc = f"⚠️ ORTA RİSK (%{risk_orani})"
        else:
            sonuc = f"✅ DÜŞÜK RİSK (%{risk_orani})"
        pdf_olustur(sonuc, risk_orani)
    except Exception as e:
        sonuc = f"Hata oluştu: {e}"

    return render_template("heart_result.html", sonuc=sonuc)


if __name__ == "__main__":
    app.run(debug=True)

