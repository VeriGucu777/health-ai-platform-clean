from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
import pickle
import shap
import os
import base64
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
FOLLOW_UP_FILE = "diabetes_follow_up.json"
HEART_FOLLOW_UP_FILE = "heart_follow_up.json"
STROKE_FOLLOW_UP_FILE = "stroke_follow_up.json"

# BURAYA EKLE ↓↓↓
def add_heart_follow_up(data, probability, result):
    follow_up_date = datetime.now() + timedelta(days=90)

    patient_record = {
        "patient_name": data.get("patient_name", "Demo Patient"),
        "disease": "Heart Disease",
        "result": result,
        "risk_probability": round(float(probability), 2),
        "created_date": datetime.now().strftime("%Y-%m-%d"),
        "follow_up_date": follow_up_date.strftime("%Y-%m-%d"),
        "doctor_note": "Heart disease risk detected. Follow-up control is recommended.",
        "status": "Waiting"
    }

    try:
        with open(HEART_FOLLOW_UP_FILE, "r", encoding="utf-8") as file:
            heart_follow_up_list = json.load(file)
    except:
        heart_follow_up_list = []

    heart_follow_up_list.append(patient_record)

    with open(HEART_FOLLOW_UP_FILE, "w", encoding="utf-8") as file:
        json.dump(heart_follow_up_list, file, indent=4, ensure_ascii=False)


# BURADAN DEVAM EDECEK ↓↓↓
def add_diabetes_follow_up(data, probability, result):

    follow_up_date = datetime.now() + timedelta(days=90)

    patient_record = {
        "patient_name": data.get("patient_name", "Demo Patient"),
        "disease": "Diabetes",
        "result": result,
        "risk_probability": round(float(probability) * 100, 2),
        "created_date": datetime.now().strftime("%Y-%m-%d"),
        "follow_up_date": follow_up_date.strftime("%Y-%m-%d"),
        "doctor_note": "High diabetes risk detected. Follow-up control is recommended.",
        "status": "Waiting"
    }

    try:
        with open(FOLLOW_UP_FILE, "r", encoding="utf-8") as file:
            follow_up_list = json.load(file)
    except:
        follow_up_list = []

    follow_up_list.append(patient_record)

    with open(FOLLOW_UP_FILE, "w", encoding="utf-8") as file:
        json.dump(follow_up_list, file, ensure_ascii=False, indent=4)


    return patient_record


def add_stroke_follow_up(data, probability, result):
    follow_up_date = datetime.now() + timedelta(days=90)

    patient_record = {
        "patient_name": data.get("patient_name", "Demo Patient"),
        "disease": "Stroke",
        "result": result,
        "risk_probability": round(float(probability), 2),
        "created_date": datetime.now().strftime("%Y-%m-%d"),
        "follow_up_date": follow_up_date.strftime("%Y-%m-%d"),
        "doctor_note": "Stroke risk detected. Follow-up control is recommended.",
        "status": "Waiting"
    }

    try:
        with open(STROKE_FOLLOW_UP_FILE, "r", encoding="utf-8") as file:
            stroke_follow_up_list = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        stroke_follow_up_list = []

    stroke_follow_up_list.append(patient_record)

    with open(STROKE_FOLLOW_UP_FILE, "w", encoding="utf-8") as file:
        json.dump(
            stroke_follow_up_list,
            file,
            ensure_ascii=False,
            indent=4
        )

    return patient_record


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({
        "status": "ok",
        "message": "Health AI Platform API is running"
    })
@app.route("/api/predict", methods=["POST"])
def api_predict():

    data = request.json

    age = data.get("age")
    glucose = data.get("glucose")
    bmi = data.get("bmi")

    sample_data = {
        "age": age,
        "glucose": glucose,
        "bmi": bmi
    }

    return jsonify({
        "success": True,
        "received_data": sample_data,
        "prediction": "High Risk"
    })
@app.route("/test")
def test_api():
    return "API is running"

def stroke_analysis_agent(data, risk_probability, result):
    explanations = []
    recommendations = []
    warning = ""

    age = data.get("age", 0)
    hypertension = data.get("hypertension", 0)
    heart_disease = data.get("heart_disease", 0)
    glucose = data.get("avg_glucose_level", 0)
    bmi = data.get("bmi", 0)
    smoking = data.get("smoking_status", "")

    if age >= 65:
        explanations.append("advanced age")
        recommendations.append("Regular blood pressure monitoring is important.")
    if hypertension == 1:
        explanations.append("hypertension")
        recommendations.append("Regular blood pressure monitoring is important.")

    if heart_disease == 1:
        explanations.append("history of heart disease")
        recommendations.append("Medical evaluation is recommended for cardiovascular health.")

    if glucose >= 140:
        explanations.append("high glucose level")
        recommendations.append("Monitoring is recommended for blood sugar/metabolic condition.")

    if bmi >= 30:
        explanations.append("high BMI")
        recommendations.append("Weight control and lifestyle monitoring are recommended due to high BMI.")

    if smoking in ["smokes", "formerly smoked"]:
        explanations.append("smoking or history of smoking")
        recommendations.append("Smoking may pose a risk to vascular health.")
    if result == "High Risk":
        warning = "🔴  Critical warning: The system has evaluated this patient as high risk."
    elif result == "Medium Risk":
        warning = "⚠️ Moderate warning: The system has evaluated this patient as medium risk."
    else:
        warning = "✅ General assessment: The system has evaluated this patient as low risk."
    if explanations:
        reason_text = ", ".join(explanations)
        explanation_sentence = f"Key factors affecting the risk assessment: {reason_text}"
    else:
        explanation_sentence = "No significant high-risk factors were identified based on the provided values."

    if recommendations:
        recommendation_text = " ".join(recommendations)
    else:
        recommendation_text = "Regular check-ups and healthy lifestyle habits are recommended from a preventive healthcare perspective."

    final_comment = (
        f"{warning} "
        f"The system evaluates the patient using both model output and clinical risk rules and classifies them in the {result.lower()} group. "
        f"The risk probability calculated by the model is {risk_probability}%. "
        f"{explanation_sentence} "
        f"{recommendation_text} "
        f"This result is not a definitive diagnosis; it is intended for educational and clinical decision support purposes only.")

    return final_comment
# Load model

model = pickle.load(open("model.pkl", "rb"))
stroke_model_path = os.path.join(BASE_DIR, "stroke", "stroke_model.pkl")
stroke_scaler_path = os.path.join(BASE_DIR, "stroke", "stroke_scaler.pkl")

stroke_model = pickle.load(open(stroke_model_path, "rb"))
stroke_scaler = pickle.load(open(stroke_scaler_path, "rb"))

explainer = shap.LinearExplainer(model, np.zeros((1, 8)))

explainer = shap.LinearExplainer(model, np.zeros((1, 8)))
heart_model_path = os.path.join(BASE_DIR, "heart", "heart_model.pkl")
heart_model = pickle.load(open(heart_model_path, "rb"))

def create_heart_pdf(result, probability, agent_comment):
    pdf_path = os.path.join(BASE_DIR, "heart_report.pdf")

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "Heart Disease Risk Analysis Report")

    c.setFont("Helvetica", 10)
    c.drawString(50, height - 100, f"Result: {result}")
    c.drawString(50, height - 130, f"Risk Percentage: %{probability}")


    grafik_yolu = os.path.join(BASE_DIR, "static", "risk_grafigi.png")
    if os.path.exists(grafik_yolu):
        grafik_y = height - 430
        c.drawImage(grafik_yolu, 50, grafik_y, width=400, height=250)

        y_position = grafik_y - 40

        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_position, "Clinical Risk Interpretation:")

        y_position -= 20
        c.setFont("Helvetica", 10)

        agent_comment = agent_comment or ""
        lines = agent_comment.split("\n")

        for line in lines:
            words = line.split(" ")
            current_line = ""

            for word in words:
                if len(current_line + word) < 75:
                    current_line += word + " "
                else:
                    c.drawString(50, y_position, current_line)
                    y_position -= 14
                    current_line = word + " "

            c.drawString(50, y_position, current_line)
            y_position -= 14
    c.save()

    print("PDF saved:", pdf_path)
    return pdf_path
def create_heart_shap(features):
    shap_path = os.path.join(BASE_DIR, "static", "heart_shap.png")

    feature_names = [
        "Age",
        "Gender",
        "Chest Pain",
        "Blood Pressure",
        "Cholesterol",
        "Blood Sugar",
        "ECG",
        "Max Heart Rate",
        "Exercise-Induced Angina",
        "ST Depression",
        "Slope",
        "Number of Vessels",
        "Thal"
    ]

    features_df = pd.DataFrame(features, columns=feature_names)

    explainer = shap.Explainer(heart_model)
    shap_values = explainer(features_df)

    plt.figure()
    shap.plots.waterfall(shap_values[0, :, 1], show=False)
    plt.tight_layout()
    plt.savefig(shap_path, bbox_inches="tight")
    plt.close()

    return shap_path

def diabetes_analysis_agent(data, probability):

    risk_percent = round(float(probability) * 100, 2)

    glucose = float(data.get("Glucose", 0))
    bmi = float(data.get("BMI", 0))
    age = int(data.get("Age", 0))
    genetic = float(data.get("DiabetesPedigreeFunction", 0))
    insulin = float(data.get("Insulin", 0))
    blood_pressure = float(data.get("BloodPressure", 0))

    factors = []
    recommendations = []

    if glucose >= 140:
        factors.append("glucose")
        recommendations.append("blood sugar monitoring")

    if bmi >= 30:
        factors.append("BMI")
        recommendations.append("weight control and diet regulation")

    if age >= 45:
        factors.append("age")
        recommendations.append("regular health check-ups")

    if genetic >= 0.8:
        factors.append("genetic predisposition")
        recommendations.append("evaluation of family history")

    if insulin > 200:
        factors.append("insulin")
        recommendations.append("insulin resistance monitoring")

    if blood_pressure >= 140:
        factors.append("blood pressure")
        recommendations.append("blood pressure monitoring")

    # Risk level
    if risk_percent < 40:
        level = "low risk"
        icon = "✅"
        summary = f"{icon} Low risk detected (%{risk_percent})"

    elif risk_percent < 70:
        level = "medium risk"
        icon = "⚠️"
        summary = f"{icon} Medium risk detected (%{risk_percent})"

    else:
        level = "high risk"
        icon = "🚨"
        summary = f"{icon} High risk detected (%{risk_percent})"

    # Top 3 most important factors (premium touch 🔥)
    if factors:
        top_factors_list = factors[:3]
        top_factors = ", ".join(top_factors_list)
        factor_text = ", ".join(factors)
    else:
        top_factors = "no significant factors"
        factor_text = "limited high-risk factors"

        # Recommendations
        if recommendations:
            recommendation_text = "\n".join(f"- {rec}" for rec in recommendations)
        else:
            recommendation_text = "- continue general health monitoring"

        return (
            f"{summary}\n\n"
            f"🧠 Key contributing factors:\n"
            f"- {top_factors}\n\n"
            f"📊 Detailed evaluation:\n"
            f"The system has classified this patient in the {level} group. "
            f"Key factors: {factor_text}.\n\n"
            f"📋 Recommended follow-up:\n"
            f"{recommendation_text}\n\n"
            f"⚠️ This result is not a definitive diagnosis; it is intended for clinical decision support purposes only."
        )

def heart_analysis_agent(data, risk_percent, result):

    factors = []
    recommendations = []

    age = int(data.get("age", 0))
    cholesterol = float(data.get("cholesterol", 0))
    max_hr = float(data.get("max_heart_rate", 0))
    oldpeak = float(data.get("oldpeak", 0))

    if age > 50:
        factors.append("age")
        recommendations.append("regular cardiology check-ups")

    if cholesterol > 240:
        factors.append("cholesterol")
        recommendations.append("cholesterol-lowering diet")

    if max_hr < 120:
        factors.append("low maximum heart rate")
        recommendations.append("exercise capacity evaluation")

    if oldpeak > 2:
        factors.append("post-exercise stress")
        recommendations.append("detailed heart test is recommended")

    if factors:
        top_factors = ", ".join(factors[:2])
        factor_text = ", ".join(factors)
    else:
        top_factors = "no significant factors"
        factor_text = "risk factors are limited"

    if recommendations:
        recommendation_text = "\n".join(f"- {r}" for r in recommendations)
    else:
        recommendation_text = "- general heart health monitoring is recommended"

    return f"""
        {result}

        Key contributing factors:
        {top_factors}

        Detailed evaluation:
        The system has evaluated this patient in terms of heart disease.
        Key factors: {factor_text}.

        Recommended follow-up:
        {recommendation_text}

        ⚠️ This result is not a definitive diagnosis; it is intended for clinical decision support purposes only.
    """

def create_pdf(result, probability, agent_comment, shap_path):

    pdf_path = "static/heart_report.pdf"

    doc = SimpleDocTemplate(pdf_path)

    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("Health AI Clinical Assistant", styles["Title"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"Result: {result}", styles["Normal"]))
    content.append(Paragraph(f"Risk Probability: %{probability}", styles["Normal"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph("Clinical Risk Interpretation:", styles["Heading2"]))
    content.append(Paragraph(agent_comment, styles["Normal"]))
    content.append(Spacer(1, 20))

    try:
        img = Image(shap_path, width=400, height=200)
        content.append(img)
    except:
        content.append(Paragraph("SHAP visualization could not be loaded", styles["Normal"]))

    doc.build(content)

    return pdf_path

def create_stroke_pdf(result, probability, agent_comment, shap_path):
    pdf_path = "static/stroke_report.pdf"

    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Stroke Prediction Report", styles["Title"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"Result: {result}", styles["Normal"]))
    content.append(Paragraph(f"Risk Probability: %{probability}", styles["Normal"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph("Clinical Risk Interpretation:", styles["Heading2"]))
    content.append(Paragraph(agent_comment, styles["Normal"]))
    content.append(Spacer(1, 20))

    try:
        img = Image(shap_path, width=400, height=200)
        content.append(img)
    except:
        content.append(Paragraph("SHAP visualization could not be loaded.", styles["Normal"]))

    doc.build(content)
    return pdf_path
@app.route("/")
def home():
    return render_template("home.html")
@app.route("/diabetes")
def diabetes_page():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # JSON or form check
    if request.is_json:
        data = request.get_json()
        model_type = data.get("model", "diabetes")
    else:
        data = request.form
        model_type = data.get("model", "heart")

        # ❤️ HEART MODEL
        # =========================

        if model_type == "heart":
            features = np.array([[
                float(data["age"]),
                float(data["gender"]),
                float(data["chest_pain"]),
                float(data["blood_pressure"]),
                float(data["cholesterol"]),
                float(data["blood_sugar"]),
                float(data["ecg"]),
                float(data["max_heart_rate"]),
                float(data["exercise_angina"]),
                float(data["st_depression"]),
                float(data["slope"]),
                float(data["num_vessels"]),
                float(data["thal"])
            ]])


        prediction = heart_model.predict(features)
        probability = heart_model.predict_proba(features)[0][1]
        create_heart_shap(features)
        risk_percent = round(probability * 100, 2)
        safe_percent = round(100 - risk_percent, 2)

        plt.figure(figsize=(6, 4))
        plt.bar(["Low Risk", "High Risk"], [safe_percent, risk_percent], color=["green", "red"])
        plt.ylim(0, 100)
        plt.ylabel("Percentage")
        #plt.title("Heart Disease Risk Chart")

        grafik_yolu = os.path.join(os.path.dirname(__file__), "heart", "static", "risk_graph.png")
        plt.savefig(grafik_yolu, bbox_inches="tight")

        plt.close()

        if risk_percent < 30:
            result = f"✅ Low Risk (%{risk_percent})"

        elif risk_percent < 60:
            result = f"⚠️ Medium Risk (%{risk_percent})"

        else:
            result = f"🚨 High Risk (%{risk_percent})"
        agent_comment = heart_analysis_agent(
            data,
            risk_percent,
            result
        )
        if not agent_comment:
            agent_comment = (
                "Based on the submitted patient data, the AI system generated a heart disease risk prediction and visualized the key contributing factors using SHAP. "
                "This interpretation helps explain which clinical features increased or decreased the predicted risk. "
                "The result is intended to support early risk awareness and should be reviewed by healthcare professionals before any medical decision."
            )
        pdf_file = create_heart_pdf(result, risk_percent, agent_comment)
        if risk_percent >= 50:
            add_heart_follow_up(request.form, risk_percent, result)
        return render_template(
            "heart_result.html",
            result=result,
            probability=risk_percent,
            agent_comment=agent_comment
        )
    # DIABETES MODEL (OLD SYSTEM)
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
        "Pregnancies",
        "Glucose",
        "Blood Pressure",
        "Skin Thickness",
        "Insulin",
        "BMI",
        "Genetic Risk",
        "Age"
    ]

    shap_result = {}

    for i in range(len(feature_names)):
        shap_result[feature_names[i]] = round(float(shap_values[0][i]), 4)

    result = "High diabetes risk" if prediction[0] == 1 else "Low diabetes risk"

    features = list(shap_result.keys())
    values = list(shap_result.values())

    # Chart
    plt.figure(figsize=(8, 5))

    colors = ['red' if v > 0 else 'blue' for v in values]

    plt.barh(features, values, color=colors)

    plt.title("SHAP Analysis (Effects)")
    plt.xlabel("Impact")
    plt.ylabel("Features")

    plt.axvline(0, color='black', linewidth=1)
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.title("SHAP Analysis")
    plt.xlabel("Impact")
    plt.ylabel("Feature")

    plt.gca().invert_yaxis()
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    if prediction[0] == 1:
        result = "Diabetes risk PRESENT"
    else:
        result = "Diabetes risk ABSENT"

    pdf_path = "report.pdf"
    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("Diabetes Prediction Report", styles["Title"]))
    content.append(Spacer(1, 12))

    content.append(Paragraph("<b>RISK RESULT</b>", styles["Heading2"]))
    content.append(Paragraph(f"<b>{result}</b>", styles["Normal"]))
    content.append(Paragraph(f"Risk Probability: %{round(probability * 100, 2)}", styles["Normal"]))

    content.append(Spacer(1, 12))
    content.append(Spacer(1, 10))

    content.append(Spacer(1, 10))
    img.seek(0)
    content.append(Paragraph("SHAP Chart:", styles["Heading2"]))
    content.append(Image(img, width=400, height=300))
    content.append(Spacer(1, 10))

    content.append(Paragraph("SHAP Values:", styles["Heading2"]))
    content.append(Spacer(1, 10))

    for key, value in shap_result.items():
        content.append(Paragraph(f"{key}: {value}", styles["Normal"]))

    doc.build(content)
    agent_comment = (
        f"The system evaluated this patient for diabetes risk. "
        f"Result: {result}. "
        f"Risk probability: {round(float(probability) * 100, 2)}%. "
        f"This result is not a definitive diagnosis; it is intended for educational and clinical decision support purposes only."
    )
    follow_up_record = None

    if prediction[0] == 1:
        follow_up_record = add_diabetes_follow_up(data, probability, result)
    return jsonify({
        "result": result,
        "probability": round(float(probability) * 100, 2),
        "shap": shap_result,
        "graph": plot_url,
        "pdf": "report.pdf",
        "agent_comment": agent_comment
    })
@app.route('/diabetes-follow-up')
def diabetes_follow_up_page():
    try:
        with open(FOLLOW_UP_FILE, "r", encoding="utf-8") as file:
            follow_up_list = json.load(file)
    except:
        follow_up_list = []

    return render_template(
        "diabetes_follow_up.html",
        follow_up_list=follow_up_list
    )
@app.route('/update-patient', methods=['POST'])
def update_patient():

    data = request.json

    with open(FOLLOW_UP_FILE, "r", encoding="utf-8") as file:
        patients = json.load(file)

    for patient in patients:

        if patient["patient_name"] == data["patient_name"]:

            patient["patient_name"] = data["new_name"]
            patient["follow_up_date"] = data["follow_up_date"]
            patient["status"] = data["status"]

            break

    with open(FOLLOW_UP_FILE, "w", encoding="utf-8") as file:
        json.dump(patients, file, indent=4, ensure_ascii=False)

    return jsonify({"success": True})
    return jsonify({"success": True})


@app.route("/delete-patient", methods=["POST"])
def delete_patient():
    data = request.get_json()

    patient_name = data.get("patient_name")
    risk_probability = data.get("risk_probability")

    try:
        risk_probability = float(risk_probability)
    except (TypeError, ValueError):
        return jsonify({
            "success": False,
            "message": "Risk probability is invalid."
        }), 400

    try:
        with open(FOLLOW_UP_FILE, "r", encoding="utf-8") as file:
            patients = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return jsonify({
            "success": False,
            "message": "Diabetes follow-up file could not be read."
        }), 404

    deleted = False
    remaining_patients = []

    for patient in patients:
        same_name = patient.get("patient_name") == patient_name

        try:
            patient_probability = float(
                patient.get("risk_probability", -1)
            )
        except (TypeError, ValueError):
            patient_probability = -1

        same_probability = patient_probability == risk_probability

        if same_name and same_probability and not deleted:
            deleted = True
            continue

        remaining_patients.append(patient)

    if not deleted:
        return jsonify({
            "success": False,
            "message": "Patient record was not found."
        }), 404

    with open(FOLLOW_UP_FILE, "w", encoding="utf-8") as file:
        json.dump(
            remaining_patients,
            file,
            indent=4,
            ensure_ascii=False
        )

    return jsonify({
        "success": True,
        "message": "Patient deleted successfully!"
    })


@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(BASE_DIR, filename)
    return send_file(file_path, as_attachment=True)

@app.route('/heart')
def heart_page():
    return render_template('heart_index.html')
@app.route('/heart-follow-up')
def heart_follow_up():

    try:
        with open(HEART_FOLLOW_UP_FILE, "r", encoding="utf-8") as file:
            follow_up_list = json.load(file)
    except:
        follow_up_list = []

    return render_template(
        "heart_follow_up.html",
        follow_up_list=follow_up_list
    )
@app.route("/delete-heart-patient", methods=["POST"])
def delete_heart_patient():
    data = request.get_json()

    patient_name = data.get("patient_name")
    risk_probability = float(data.get("risk_probability"))

    try:
        with open(HEART_FOLLOW_UP_FILE, "r", encoding="utf-8") as file:
            patients = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return jsonify({
            "success": False,
            "message": "Heart follow-up file could not be read."
        }), 404

    deleted = False
    updated_patients = []

    for patient in patients:
        same_name = patient.get("patient_name") == patient_name
        same_probability = float(
            patient.get("risk_probability", -1)
        ) == risk_probability

        if same_name and same_probability and not deleted:
            deleted = True
            continue

        updated_patients.append(patient)

    if not deleted:
        return jsonify({
            "success": False,
            "message": "Patient record was not found."
        }), 404

    with open(HEART_FOLLOW_UP_FILE, "w", encoding="utf-8") as file:
        json.dump(
            updated_patients,
            file,
            indent=4,
            ensure_ascii=False
        )

    return jsonify({
        "success": True,
        "message": "Patient deleted successfully!"
    })
@app.route("/download-heart-pdf")
def download_heart_pdf():
    pdf_path = os.path.join(BASE_DIR, "heart_report.pdf")
    return send_file(pdf_path, as_attachment=True)

@app.route("/stroke")
def stroke_page():
    return render_template("stroke_index.html")
@app.route('/stroke-follow-up')
def stroke_follow_up():
    try:
        with open(STROKE_FOLLOW_UP_FILE, "r", encoding="utf-8") as file:
            follow_up_list = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        follow_up_list = []

    return render_template(
        "stroke_follow_up.html",
        follow_up_list=follow_up_list
    )

@app.route('/stroke_predict', methods=['POST'])
def stroke_predict():
    try:


        # FİLE PATHS
        stroke_model_path = os.path.join(BASE_DIR, "stroke", "stroke_model.pkl")
        stroke_scaler_path = os.path.join(BASE_DIR, "stroke", "stroke_scaler.pkl")

        # LOAD MODEL AND SCALAR
        stroke_model = pickle.load(open(stroke_model_path, "rb"))
        stroke_scaler = pickle.load(open(stroke_scaler_path, "rb"))

        gender_map = {
            "Erkek": "Male",
            "Kadın": "Female"
        }

        married_map = {
            "Evet": "Yes",
            "Hayır": "No"
        }

        work_map = {
            "Devlet": "Govt_job",
            "Özel Sektör": "Private",
            "Kendi İşi": "Self-employed",
            "Çalışmıyor": "Never_worked"
        }

        residence_map = {
            "Şehir": "Urban",
            "Kırsal": "Rural"
        }

        smoking_map = {
            "İçiyor": "smokes",
            "Hiç içmedi": "never smoked",
            "Bıraktı": "formerly smoked"
        }
        # FORM DATA
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])

        gender = request.form.get('gender')
        ever_married = request.form.get('ever_married')
        work_type = request.form.get('work_type')
        residence_type = request.form.get('residence_type')
        smoking_status = request.form.get('smoking_status')

        # 16 FEATURES EXPECTED BY THE MODEL
        row = {
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "avg_glucose_level": avg_glucose_level,
            "bmi": bmi,

            "gender_Male": 1 if gender == "Male" else 0,
            "gender_Other": 1 if gender == "Other" else 0,

            "ever_married_Yes": 1 if ever_married == "Yes" else 0,

            "work_type_Never_worked": 1 if work_type == "Never_worked" else 0,
            "work_type_Private": 1 if work_type == "Private" else 0,
            "work_type_Self-employed": 1 if work_type == "Self-employed" else 0,
            "work_type_children": 1 if work_type == "children" else 0,

            "Residence_type_Urban": 1 if residence_type == "Urban" else 0,

            "smoking_status_formerly smoked": 1 if smoking_status == "formerly smoked" else 0,
            "smoking_status_never smoked": 1 if smoking_status == "never smoked" else 0,
            "smoking_status_smokes": 1 if smoking_status == "smokes" else 0
        }
        print("GELEN VERİLER:")
        print("age:", age)
        print("hypertension:", hypertension)
        print("heart_disease:", heart_disease)
        print("avg_glucose_level:", avg_glucose_level)
        print("bmi:", bmi)
        print("gender:", gender)
        print("ever_married:", ever_married)
        print("work_type:", work_type)
        print("residence_type:", residence_type)
        print("smoking_status:", smoking_status)
        print("ROW:", row)
        print("FEATURE COUNT:", len(row))
        final_input = pd.DataFrame([row])

        # SCALİNG
        final_input_scaled = stroke_scaler.transform(final_input)

        # PREDİCTİON
        probability = stroke_model.predict_proba(final_input_scaled)[0][1] * 100
        print("MODEL OLASILIK:", probability)
        print("MODEL TAHMİNİ:", stroke_model.predict(final_input_scaled)[0])
        print("GELEN INPUT:", final_input)
        print("SCALE EDİLMİŞ INPUT:", final_input_scaled)

        # SHAP ANALYSIS - FOR LOGİSTİC REGRESSİON
        shap_path = os.path.join("static", "shap_plot.png")

        feature_names_en = {
            "age": "Age",
            "hypertension": "Hypertension",
            "heart_disease": "Heart Disease",
            "avg_glucose_level": "Average Glucose",
            "bmi": "Body Mass Index",
            "gender_Male": "Male",
            "gender_Other": "Other",
            "ever_married_Yes": "Married",
            "work_type_Never_worked": "Never Worked",
            "work_type_Private": "Private Sector",
            "work_type_Self-employed": "Self-employed",
            "work_type_children": "Child",
            "Residence_type_Urban": "Urban",
            "smoking_status_formerly smoked": "Former Smoker",
            "smoking_status_never smoked": "Never Smoked",
            "smoking_status_smokes": "Currently Smokes"
        }
        explainer = shap.LinearExplainer(stroke_model, final_input_scaled)
        shap_values = explainer.shap_values(final_input_scaled)

       # SHAP ANALYSIS (REAL MEANING)
        explainer = shap.LinearExplainer(stroke_model, final_input_scaled)
        shap_values = explainer.shap_values(final_input_scaled)

        import numpy as np

        # SORT THE MOST İMPORTANT FEATURES
        idx = np.argsort(np.abs(shap_values[0]))[::-1]

        plt.figure(figsize=(10, 5))

        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0][idx],
                base_values=explainer.expected_value,
                data=final_input.iloc[0].values[idx],
                feature_names=final_input.columns[idx]
            ),
            show=False
        )

        plt.savefig(shap_path, bbox_inches='tight')
        plt.close()

        # Risk score başlangıç
        features_scaled = stroke_scaler.transform(final_input)
        probability = stroke_model.predict_proba(features_scaled)[0][1] * 100
        risk_score = min(max(probability, 0), 100)

        # CLİNİCAL FACTORS CURRENTLY DİSABLED

        #if hypertension == 1:
            #risk_score += 10

        #if heart_disease == 1:
            #risk_score += 15

        #if smoking_status in ["smokes", "formerly smoked"]:
            #risk_score += 10

        #if bmi > 30:
            #risk_score += 10

        #if age >= 65:
             #risk_score += 15

        # Final decision
        if risk_score < 20:
            result_text = "Low Risk"
        elif risk_score < 50:
            result_text = "Medium Risk"
        else:
            result_text = "High Risk"

        patient_data = {
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "avg_glucose_level": avg_glucose_level,
            "bmi": bmi,
            "smoking_status": smoking_status
        }

        agent_comment = stroke_analysis_agent(
            patient_data,
            round(risk_score, 2),
            result_text
        )

        pdf_path = create_stroke_pdf(
            result_text,
            round(risk_score, 2),
            agent_comment,
            None
        )
        add_stroke_follow_up(
            request.form,
            round(risk_score, 2),
            result_text
        )
        return render_template(
        'stroke_result.html',
        result=result_text,
        probability=round(risk_score, 2),
        agent_comment=agent_comment,
        shap_plot=shap_path
    )

    except Exception as e:
        return f"An error occurred: {e}"

@app.route("/delete-stroke-patient", methods=["POST"])
def delete_stroke_patient():
    data = request.get_json()

    patient_name = data.get("patient_name")
    risk_probability = float(data.get("risk_probability"))

    try:
        with open(STROKE_FOLLOW_UP_FILE, "r", encoding="utf-8") as file:
            stroke_follow_up_list = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return jsonify({
            "success": False,
            "message": "Patient list could not be loaded."
        }), 404

    original_count = len(stroke_follow_up_list)

    stroke_follow_up_list = [
        patient for patient in stroke_follow_up_list
        if not (
            patient.get("patient_name") == patient_name
            and float(patient.get("risk_probability", 0)) == risk_probability
        )
    ]

    if len(stroke_follow_up_list) == original_count:
        return jsonify({
            "success": False,
            "message": "Patient could not be found."
        }), 404

    with open(STROKE_FOLLOW_UP_FILE, "w", encoding="utf-8") as file:
        json.dump(
            stroke_follow_up_list,
            file,
            ensure_ascii=False,
            indent=4
        )

    return jsonify({
        "success": True,
        "message": "Patient deleted successfully!"
    })
if __name__ == "__main__":
    app.run(debug=True)