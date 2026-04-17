from flask import Flask, render_template, request, redirect, url_for, session, make_response, jsonify
from openai import OpenAI
from werkzeug.security import generate_password_hash, check_password_hash
from fpdf import FPDF
import sqlite3
import json
import os
import PyPDF2
import base64
import mimetypes

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMERGENCY_KEYWORDS = [
    "chest pain",
    "difficulty breathing",
    "shortness of breath",
    "unconscious",
    "severe bleeding",
    "blood vomiting",
    "seizure",
    "fainting"
]


def init_chat_state():
    if "chat_messages" not in session:
        session["chat_messages"] = [
            {
                "role": "system",
                "content": (
                    "You are a compassionate AI health assistant speaking like a doctor in a one-on-one chat. "
                    "Ask only one relevant follow-up question at a time based on the user's latest reply. "
                    "Keep the tone warm, human, calm, and supportive. "
                    "You can discuss any symptom, illness, or health concern the user mentions. "
                    "You are not limited to a fixed disease list. "
                    "Once you have enough information, provide: "
                    "likely condition or possible explanation, why it may match, basic care suggestions, diet advice, "
                    "light recovery guidance if appropriate, and warning signs for seeing a real doctor. "
                    "Do not claim to be a licensed doctor. "
                    "Do not give a guaranteed final diagnosis. "
                    "If symptoms sound urgent, advise immediate medical attention. "
                    "Keep replies concise and conversational."
                )
            }
        ]


def get_emergency_reply(user_message):
    text = user_message.lower()
    for keyword in EMERGENCY_KEYWORDS:
        if keyword in text:
            return (
                "Your symptoms may need urgent medical attention. "
                "Please contact a doctor or emergency care as soon as possible."
            )
    return None


def get_ai_chat_reply(user_message, user_name):
    try:
        init_chat_state()

        chat_messages = session["chat_messages"]

        chat_messages.append({
            "role": "user",
            "content": f"User name: {user_name}\nMessage: {user_message}"
        })

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=chat_messages,
            temperature=0.7
        )

        reply = response.choices[0].message.content.strip()

        chat_messages.append({
            "role": "assistant",
            "content": reply
        })

        session["chat_messages"] = chat_messages
        return reply

    except Exception as e:
        print("CHAT ERROR:", e)
        return (
            f"Hello {user_name}, I’m sorry — I couldn’t continue the conversation right now. "
            f"Please try again in a moment."
        )

def get_followup_questions(symptoms):
    text = symptoms.lower()
    questions = []

    if "fever" in text:
        questions.append("Do you have fever? Since when?")
    if "cough" in text:
        questions.append("Is the cough dry or with mucus?")
    if "headache" in text:
        questions.append("Is the headache mild or severe?")
    if "stomach" in text or "abdominal" in text:
        questions.append("Do you also have vomiting or loose motions?")
    if "chest pain" in text:
        questions.append("Does the pain spread to the arm, back, or jaw?")
    if "breathing" in text or "shortness of breath" in text:
        questions.append("Are you feeling difficulty while walking or even at rest?")

    if not questions:
        questions = [
            "Since when are you having these symptoms?",
            "Are the symptoms getting better or worse?",
            "Do you have any pain, weakness, or fever with this?"
        ]

    return questions


DB_NAME = "users.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            symptoms TEXT,
            possible_condition TEXT,
            severity TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    conn.commit()
    conn.close()


def check_emergency(symptoms):
    text = symptoms.lower()
    for word in EMERGENCY_KEYWORDS:
        if word in text:
            return True
    return False


def get_ai_health_response(symptoms, user_name):
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a healthcare assistant for educational purposes only. "
                        f"Address the user by name: {user_name}. "
                        f"Do not give a final diagnosis. "
                        f"Return only valid JSON with these exact keys: "
                        f"possible_condition, reason, severity, basic_care, see_doctor_when, disclaimer. "
                        f"Severity must be one of: Low, Medium, High. "
                        f"Keep every field short, simple, and clear."
                    )
                },
                {
                    "role": "user",
                    "content": f"My symptoms are: {symptoms}"
                }
            ]
        )

        content = response.choices[0].message.content
        return json.loads(content)

    except Exception:
        return {
            "possible_condition": "Unable to analyze right now",
            "reason": "The AI service could not process the symptoms at the moment.",
            "severity": "Medium",
            "basic_care": "Rest, stay hydrated, and monitor symptoms.",
            "see_doctor_when": "See a doctor if symptoms worsen or continue.",
            "disclaimer": f"This is not a final medical diagnosis, {user_name}."
        }
    

def get_ai_report_summary(report_text, user_name):
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a medical report summarizer for educational purposes only. "
                        f"Address the user by name: {user_name}. "
                        f"Summarize the report in very simple English. "
                        f"Do not give a final diagnosis. "
                        f"Keep the answer short, clear, and easy to understand."
                    )
                },
                {
                    "role": "user",
                    "content": f"Summarize this medical report:\n\n{report_text}"
                }
            ]
        )

        return response.choices[0].message.content

    except Exception:
        return (
            f"Hello {user_name}. I could not analyze the report with AI right now. "
            f"Please check the uploaded text and try again."
        )
    
def get_ai_image_report_summary(image_bytes, filename, user_name):
    try:
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type is None:
            mime_type = "image/png"

        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{base64_image}"

        response = client.responses.create(
            model="gpt-5.4-mini",
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                f"You are a medical report summarizer for educational purposes only. "
                                f"Address the user by name: {user_name}. "
                                f"Read the uploaded medical report image carefully and summarize it in very simple English. "
                                f"Do not give a final diagnosis. "
                                f"Mention key findings only."
                            )
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Please analyze this medical report image and summarize the important findings."
                        },
                        {
                            "type": "input_image",
                            "image_url": data_url
                        }
                    ]
                }
            ]
        )

        return response.output_text

    except Exception:
        return (
            f"Hello {user_name}. I could not analyze the image report right now. "
            f"Please upload a clearer image or try again."
        )

def get_user_history(user_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT symptoms, possible_condition, severity
        FROM history
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 5
    """, (user_id,))

    rows = cursor.fetchall()
    conn.close()

    history = []
    for row in rows:
        history.append({
            "symptoms": row[0],
            "condition": row[1],
            "severity": row[2]
        })
    return history


@app.route("/")
def home():
    if "user_id" not in session:
        return redirect(url_for("login"))

    history = get_user_history(session["user_id"])
    return render_template(
        "index.html",
        user_name=session["user_name"],
        history=history
    )


@app.route("/signup", methods=["GET", "POST"])
def signup():
    message = ""

    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]

        hashed_password = generate_password_hash(password)

        try:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                (name, email, hashed_password)
            )
            conn.commit()
            conn.close()

            return redirect(url_for("login"))

        except sqlite3.IntegrityError:
            message = "Email already exists. Please login."

    return render_template("signup.html", message=message)


@app.route("/login", methods=["GET", "POST"])
def login():
    message = ""

    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, password FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session["user_id"] = user[0]
            session["user_name"] = user[1]
            return redirect(url_for("home"))
        else:
            message = "Invalid email or password."

    return render_template("login.html", message=message)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/predict", methods=["POST"])
def predict():
    if "user_id" not in session:
        return redirect(url_for("login"))

    symptoms = request.form["symptoms"]
    emergency = check_emergency(symptoms)
    result = get_ai_health_response(symptoms, session["user_name"])

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO history (user_id, symptoms, possible_condition, severity)
        VALUES (?, ?, ?, ?)
    """, (
        session["user_id"],
        symptoms,
        result["possible_condition"],
        result["severity"]
    ))
    conn.commit()
    conn.close()

    history = get_user_history(session["user_id"])

    return render_template(
        "index.html",
        user_name=session["user_name"],
        user_input=symptoms,
        result=result,
        emergency=emergency,
        history=history
    )


@app.route("/download-report", methods=["POST"])
def download_report():
    if "user_id" not in session:
        return redirect(url_for("login"))

    symptoms = request.form["symptoms"]
    possible_condition = request.form["possible_condition"]
    reason = request.form["reason"]
    severity = request.form["severity"]
    basic_care = request.form["basic_care"]
    see_doctor_when = request.form["see_doctor_when"]
    disclaimer = request.form["disclaimer"]

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "AI Health Assistant Report", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "", 12)

    pdf.multi_cell(0, 10, f"User: {session['user_name']}")
    pdf.multi_cell(0, 10, f"Symptoms Entered: {symptoms}")
    pdf.multi_cell(0, 10, f"Possible Condition: {possible_condition}")
    pdf.multi_cell(0, 10, f"Why This May Match: {reason}")
    pdf.multi_cell(0, 10, f"Severity Level: {severity}")
    pdf.multi_cell(0, 10, f"Basic Care: {basic_care}")
    pdf.multi_cell(0, 10, f"When to See a Doctor: {see_doctor_when}")
    pdf.multi_cell(0, 10, f"Medical Disclaimer: {disclaimer}")

    pdf_output = pdf.output(dest="S").encode("latin-1")

    response = make_response(pdf_output)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = "attachment; filename=health_report.pdf"
    return response
   


@app.route("/advanced")
def advanced():
    if "user_id" not in session:
        return redirect(url_for("login"))

    session.pop("chat_messages", None)
    return render_template("advanced.html")

@app.route("/chat", methods=["POST"])
def chat():
    if "user_id" not in session:
        return jsonify({"reply": "Please login first."}), 401

    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"reply": "Please type your message."}), 400

    emergency_reply = get_emergency_reply(user_message)
    if emergency_reply:
        return jsonify({"reply": emergency_reply})

    reply = get_ai_chat_reply(user_message, session["user_name"])
    return jsonify({"reply": reply})

@app.route("/reset_chat", methods=["POST"])
def reset_chat():
    if "user_id" not in session:
        return jsonify({"message": "Please login first."}), 401

    session.pop("chat_messages", None)
    return jsonify({"message": "Chat reset successful"})



@app.route("/advanced-predict", methods=["POST"])
def advanced_predict():
    if "user_id" not in session:
        return redirect(url_for("login"))

    symptoms = request.form["symptoms"]
    emergency = check_emergency(symptoms)
    result = get_ai_health_response(symptoms, session["user_name"])
    followup_questions = get_followup_questions(symptoms)

    return render_template(
        "advanced.html",
        user_input=symptoms,
        result=result,
        emergency=emergency,
        followup_questions=followup_questions,
        report_summary=None
    )



@app.route("/upload-report", methods=["POST"])
def upload_report():
    if "user_id" not in session:
        return jsonify({"reply": "Please login first."}), 401

    uploaded_file = request.files.get("report_file")

    if not uploaded_file or uploaded_file.filename == "":
        return jsonify({"reply": "No file selected."})

    filename = uploaded_file.filename.lower()

    try:
        if filename.endswith(".txt"):
            report_text = uploaded_file.read().decode("utf-8")
            if not report_text.strip():
                return jsonify({"reply": "The uploaded text file is empty."})

            report_summary = get_ai_report_summary(report_text, session["user_name"])

        elif filename.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            report_text = ""

            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    report_text += page_text + "\n"

            if not report_text.strip():
                return jsonify({"reply": "No readable text was found in this PDF. It may be a scanned image PDF."})

            report_summary = get_ai_report_summary(report_text, session["user_name"])

        elif filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_bytes = uploaded_file.read()

            if not image_bytes:
                return jsonify({"reply": "The uploaded image file is empty."})

            report_summary = get_ai_image_report_summary(
                image_bytes,
                uploaded_file.filename,
                session["user_name"]
            )

        else:
            return jsonify({"reply": "Please upload only a .txt, .pdf, .jpg, .jpeg, or .png medical report file."})

        return jsonify({"summary": report_summary})

    except Exception:
        return jsonify({"reply": "Unable to read the file. Please upload a valid text, PDF, or clear image report."})


if __name__ == "__main__":
    init_db()
    app.run(debug=True)

