from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response
import os
import random
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from io import BytesIO
from xhtml2pdf import pisa

app = Flask(__name__)

# ---------- ML Model Training ----------
def train_model():
    df = pd.read_csv("career_data.csv")

    le_interest = LabelEncoder()
    le_skills = LabelEncoder()
    le_mood = LabelEncoder()
    le_career = LabelEncoder()

    df["interest_encoded"] = le_interest.fit_transform(df["interest"])
    df["skills_encoded"] = le_skills.fit_transform(df["skills"])
    df["mood_encoded"] = le_mood.fit_transform(df["mood"])
    df["career_encoded"] = le_career.fit_transform(df["career"])

    X = df[["age", "interest_encoded", "skills_encoded", "mood_encoded"]]
    y = df["career_encoded"]

    model = RandomForestClassifier()
    model.fit(X, y)

    return model, le_interest, le_skills, le_mood, le_career

model, le_interest, le_skills, le_mood, le_career = train_model()

# ---------- Folder Setup ----------
if not os.path.exists("data"):
    os.mkdir("data")

# ---------- Routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        name = request.form["name"]
        age = request.form["age"]
        interest = request.form["interest"]
        mood = request.form.get("mood", "Unknown")
        skills = request.form.getlist("skills")

        # ML Prediction
        interest_encoded = le_interest.transform([interest])[0]
        skill_encoded = le_skills.transform([skills[0]])[0] if skills else 0
        mood_encoded = le_mood.transform([mood])[0]

        X_new = [[int(age), interest_encoded, skill_encoded, mood_encoded]]
        y_pred = model.predict(X_new)
        suggestion = le_career.inverse_transform(y_pred)[0]

        with open("data/user_data.txt", "a") as file:
            file.write(f"{name},{age},{interest},{'/'.join(skills)},{mood},{suggestion}\n")

        return redirect(url_for("dashboard"))

    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    user_entries = []
    mood_count = {"Happy": 0, "Sad": 0, "Neutral": 0}
    career_count = Counter()

    quotes = {
        "Happy": [
            "Keep spreading your joy! 😊",
            "Happiness looks good on you!",
            "Stay positive and keep shining!"
        ],
        "Sad": [
            "It's okay to not be okay. Better days are ahead 💙",
            "You're stronger than you think.",
            "This too shall pass. Keep going!"
        ],
        "Neutral": [
            "A small step today is progress!",
            "Stay focused and keep moving.",
            "Balance is the key to everything."
        ]
    }

    try:
        with open("data/user_data.txt", "r") as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split(",")
                if len(parts) == 6:
                    entry = {
                        "name": parts[0],
                        "age": parts[1],
                        "interest": parts[2],
                        "skills": parts[3],
                        "mood": parts[4],
                        "suggestion": parts[5]
                    }
                    user_entries.append(entry)

                    mood = parts[4]
                    if mood in mood_count:
                        mood_count[mood] += 1

                    career = parts[5]
                    career_count[career] += 1

    except FileNotFoundError:
        user_entries = []

    if user_entries:
        last_mood = user_entries[-1]["mood"]
        quote_for_mood = random.choice(quotes.get(last_mood, ["Keep going, you are doing great!"]))
    else:
        quote_for_mood = "Welcome to LifePathAI! Let's begin your journey."

    return render_template("dashboard.html", entries=user_entries, mood_count=mood_count, career_count=dict(career_count), quote=quote_for_mood)

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.get_json().get("message", "").lower()

    if "career" in user_message:
        reply = "Let's talk about your skills and interests. What's your strongest skill?"
    elif "confused" in user_message or "don't know" in user_message:
        reply = "It's okay to feel confused. Start with what you enjoy doing!"
    elif "ai" in user_message:
        reply = "AI is booming! You could become an AI developer or data scientist."
    elif "sad" in user_message or "stress" in user_message:
        reply = "Breathe deeply. You're stronger than you think. Want a motivational quote?"
    elif "hello" in user_message or "hi" in user_message:
        reply = "Hello! I'm your AI buddy. Ask me anything about career or motivation!"
    else:
        reply = "That's interesting! Can you tell me more?"

    return jsonify({"reply": reply})

@app.route("/download/<int:index>")
def download(index):
    try:
        with open("data/user_data.txt", "r") as file:
            lines = file.readlines()
            if index < len(lines):
                parts = lines[index].strip().split(",")
                name, age, interest, skills, mood, suggestion = parts

                quote_list = {
                    "Happy": ["Keep spreading your joy! 😊", "Happiness looks good on you!", "Stay positive and keep shining!"],
                    "Sad": ["It's okay to not be okay. Better days are ahead 💙", "You're stronger than you think.", "This too shall pass. Keep going!"],
                    "Neutral": ["A small step today is progress!", "Stay focused and keep moving.", "Balance is the key to everything."]
                }

                quote = random.choice(quote_list.get(mood, ["Keep going!"]))

                rendered = render_template("pdf_template.html",
                                           name=name,
                                           age=age,
                                           interest=interest,
                                           skills=skills,
                                           mood=mood,
                                           suggestion=suggestion,
                                           quote=quote)

                pdf = BytesIO()
                pisa.CreatePDF(BytesIO(rendered.encode("utf-8")), dest=pdf)
                response = make_response(pdf.getvalue())
                response.headers["Content-Type"] = "application/pdf"
                response.headers["Content-Disposition"] = f"attachment; filename={name}_career_report.pdf"
                return response
    except:
        return "Error generating PDF."
if __name__ == "__main__":
    app.run(debug=True)
