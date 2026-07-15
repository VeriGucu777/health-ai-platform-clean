from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Home page
@app.route("/")
def home():
    return render_template("heart_index.html")


# Cleaning function
def clean(x):
    try:
        return float(x)
    except:
        return 0


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = clean(request.form.get("yas"))
        gender = clean(request.form.get("cinsiyet"))
        chest_pain = clean(request.form.get("gogus_agrisi"))
        blood_pressure = clean(request.form.get("tansiyon"))
        cholesterol = clean(request.form.get("kolesterol"))
        blood_sugar = clean(request.form.get("kan_sekeri"))
        ekg = clean(request.form.get("ekg"))
        max_heart_rate = clean(request.form.get("max_nabiz"))
        exercise_angina = clean(request.form.get("egzersiz_anlina"))
        st_depression = clean(request.form.get("st_depresyon"))
        slope = clean(request.form.get("egim"))
        vessel_count = clean(request.form.get("damar_sayisi"))
        thal = clean(request.form.get("thal"))

        data = np.array([[age, gender, chest_pain, blood_pressure, cholesterol,
                          blood_sugar, ekg, max_heart_rate, exercise_angina,
                          st_depression, slope, vessel_count, thal]])

        print("DATA:", data)

        probability = model.predict_proba(data)[0]
        print("PROBABILITY:", probability)

        # threshold
        risk_rate = round(probability[1] * 100, 2)

        if probability[1] > 0.70:
            result = f"🚨 HIGH RISK (%{risk_rate})"
        elif probability[1] > 0.30:
            result = f"⚠️ MEDIUM RISK (%{risk_rate})"
        else:
            result = f"✅ LOW RISK (%{risk_rate})"

    except Exception as e:
        result = f"An error occurred: {e}"

    return render_template("heart_result.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)