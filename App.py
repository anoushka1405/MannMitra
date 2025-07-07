from flask import Flask, render_template, request, jsonify, session
import os
from dotenv import load_dotenv
import google.generativeai as genai
from Aasha_chatbot import first_message, continue_convo, get_emotion_label


# Load env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print("🔑 Google API Key loaded is:", GOOGLE_API_KEY)

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

app = Flask(__name__)
app.secret_key = "aasha-is-kind"

@app.route("/")
def home():
    return render_template("indexnew.html")

@app.route("/about")
def about():
    return render_template("about2.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/services")
def services():
    return render_template("services2.html")

@app.route("/privacypolicy")
def privacypolicy():
    return render_template("privacypolicy.html")

@app.route("/termsofuse")
def termsofuse():
    return render_template("termsofuse.html")

@app.route("/get", methods=["POST"])
def chat():
    user_message = request.form["msg"]
    emotion = get_emotion_label(user_message)

    try:
        # Detect if this is the first message in the chat
        if "chat_started" not in session:
            session["chat_started"] = True
            reply = first_message(user_message)
        else:
            reply = continue_convo(user_message)

        return jsonify({
            "reply": reply,
            "emotion": emotion
        })

    except Exception as e:
        print("💥 Error:", e)
        return jsonify({
            "reply": "Oops, I’m having trouble replying right now. Please try again later.",
            "emotion": "neutral"
        })

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5050)



