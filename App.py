from flask import Flask, render_template, request, jsonify, session
import os
from dotenv import load_dotenv
import google.generativeai as genai
from Aasha_chatbot import first_message, continue_convo, get_emotion_label, is_exit_intent

# Load env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
#print("🔑 Google API Key loaded is:", GOOGLE_API_KEY)

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
    exit_intent = is_exit_intent(user_message)

    try:
        if "chat_started" not in session:
            session["chat_started"] = True
            reply, meta = first_message(user_message)
        else:
            if exit_intent:
                reply = "I'm really glad we talked today. Thank you for visiting <strong>Mann Mitra</strong>Please take care 💙"
                return jsonify({
                    "reply": reply,
                    "emotion": "neutral",
                    "exit_intent": True,
                    "celebration_type": None
                })
            else:
                reply, meta = continue_convo(user_message)

        return jsonify({
            "reply": reply,
            "emotion": meta["emotion"],
            "exit_intent": exit_intent,
            "celebration_type": meta["celebration_type"]
        })

    except Exception as e:
        print("💥 Error:", e)
        return jsonify({
            "reply": "Oops, I’m having trouble replying right now. Please try again later.",
            "emotion": "neutral",
            "exit_intent": False,
            "celebration_type": None
        })
if __name__ == "__main__":
    app.run(debug=True)


