from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
import pickle
import shap
import os
import base64

import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

# Modeli yükle
model = pickle.load(open("model.pkl", "rb"))
explainer = shap.LinearExplainer(model, np.zeros((1, 8)))
heart_model = pickle.load(open("heart\heart_model.pkl", "rb"))

def create_heart_pdf(result, probability):
    pdf_path = os.path.join(BASE_DIR, "heart_report.pdf")

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "Kalp Hastaligi Risk Analizi Raporu")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, f"Sonuc: {result}")
    c.drawString(50, height - 130, f"Risk Yuzdesi: %{probability}")

    grafik_yolu = os.path.join(BASE_DIR, "static", "risk_grafigi.png")
    if os.path.exists(grafik_yolu):
        c.drawImage(grafik_yolu, 50, height - 430, width=400, height=250)

    c.save()

    print("PDF kaydedildi:", pdf_path)
    return pdf_path
def create_heart_shap(features):
    shap_path = os.path.join(BASE_DIR, "static", "heart_shap.png")

    explainer = shap.Explainer(heart_model)
    shap_values = explainer(features)

    plt.figure()
    shap.plots.waterfall(shap_values[0, :, 1], show=False)
    plt.tight_layout()
    plt.savefig(shap_path, bbox_inches="tight")
    plt.close()

    return shap_path
@app.route("/")
def home():
    return render_template("home.html")
@app.route("/diabetes")
def diabetes_page():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # JSON mu form mu kontrolü
    if request.is_json:
        data = request.get_json()
        model_type = data.get("model", "diabetes")
    else:
        data = request.form
        model_type = data.get("model", "heart")

    # =========================
    # ❤️ KALP MODELİ
    # =========================
    if model_type == "heart":
        features = np.array([[
            float(data["yas"]),
            float(data["cinsiyet"]),
            float(data["gogus_agrisi"]),
            float(data["tansiyon"]),
            float(data["kolesterol"]),
            float(data["kan_sekeri"]),
            float(data["ekg"]),
            float(data["max_nabiz"]),
            float(data["egzersiz_anjina"]),
            float(data["st_depresyon"]),
            float(data["egim"]),
            float(data["damar_sayisi"]),
            float(data["thal"])
        ]])

        prediction = heart_model.predict(features)
        probability = heart_model.predict_proba(features)[0][1]
        create_heart_shap(features)
        risk_percent = round(probability * 100, 2)
        safe_percent = round(100 - risk_percent, 2)


        plt.figure(figsize=(6, 4))
        plt.bar(["Düşük Risk", "Yüksek Risk"], [safe_percent, risk_percent], color=["green", "red"])
        plt.ylim(0, 100)
        plt.ylabel("Yüzde")
        plt.title("Kalp Hastalığı Risk Grafiği")

        grafik_yolu = os.path.join(os.path.dirname(__file__), "heart", "static", "risk_grafigi.png")
        plt.savefig(grafik_yolu, bbox_inches="tight")

        plt.close()
        if risk_percent < 40:
            result = f"✅ Düşük Risk (%{risk_percent})"
        elif risk_percent < 70:
            result = f"⚠️ Orta Risk (%{risk_percent})"
        else:
            result = f"🚨 Yüksek Risk (%{risk_percent})"
        pdf_file = create_heart_pdf(result, risk_percent)
        return render_template(
            "heart_result.html",
            result=result,
            probability=risk_percent
        )
    # =========================
    # 🧪 DİYABET MODELİ (ESKİ SİSTEM)
    # =========================
    features = np.array([
        data["Pregnancies"],
        data["Glucose"],
        data["BloodPressure"],
        data["SkinThickness"],
        data["Insulin"],
        data["BMI"],
        data["DiabetesPedigreeFunction"],
        data["Age"]
    ]).reshape(1, -1)

    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]

    shap_values = explainer.shap_values(features)

    feature_names = [
        "Gebelik",
        "Glikoz",
        "Tansiyon",
        "Deri Kalınlığı",
        "İnsülin",
        "BMI",
        "Genetik",
        "Yaş"
    ]

    shap_result = {}

    for i in range(len(feature_names)):
        shap_result[feature_names[i]] = round(float(shap_values[0][i]), 4)

    result = "Diyabet riski yüksek" if prediction[0] == 1 else "Diyabet riski düşük"

    features = list(shap_result.keys())
    values = list(shap_result.values())

    # Grafik
    plt.figure(figsize=(8, 5))

    colors = ['red' if v > 0 else 'blue' for v in values]

    plt.barh(features, values, color=colors)

    plt.title("SHAP Analizi (Etkiler)")
    plt.xlabel("Etkisi")
    plt.ylabel("Özellikler")

    plt.axvline(0, color='black', linewidth=1)
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.title("SHAP Analizi")
    plt.xlabel("Etki")
    plt.ylabel("Özellik")

    plt.gca().invert_yaxis()
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    if prediction[0] == 1:
       result = "Diyabet riski VAR"
    else:
       result = "Diyabet riski YOK"
    pdf_path = "report.pdf"
    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("Diyabet Tahmin Raporu", styles["Title"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"Sonuç: {result}", styles["Normal"]))
    content.append(Paragraph(f"Olasılık: %{round(probability * 100, 2)}", styles["Normal"]))
    content.append(Spacer(1, 10))
    img.seek(0)
    content.append(Paragraph("SHAP Grafik:", styles["Heading2"]))
    content.append(Image(img, width=400, height=300))
    content.append(Spacer(1, 10))

    content.append(Paragraph("SHAP Değerleri:", styles["Heading2"]))
    content.append(Spacer(1, 10))

    for key, value in shap_result.items():
      content.append(Paragraph(f"{key}: {value}", styles["Normal"]))

    doc.build(content)
    return jsonify({
        "result": result,
        "probability": round(float(probability) * 100, 2),
        "shap": shap_result,
        "graph": plot_url,
        "pdf": "report.pdf"
    })

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(os.getcwd(), filename)
    return send_file(file_path, as_attachment=True)

@app.route('/heart')
def heart_page():
    return render_template('heart_index.html')

@app.route("/download-heart-pdf")
def download_heart_pdf():
    pdf_path = os.path.join(BASE_DIR, "heart_report.pdf")
    return send_file(pdf_path, as_attachment=True)
if __name__ == "__main__":
    app.run(debug=True)