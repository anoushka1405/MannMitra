from flask import Flask, render_template, request, jsonify, session, make_response
import os
from dotenv import load_dotenv
import google.generativeai as genai
from Aasha_chatbot import first_message, continue_convo, is_exit_intent

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Flask setup
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "aasha-is-kind")

@app.route("/")
def home():
    return render_template("index.html")

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
    user_message = request.json.get("msg", "")

    GROUNDING_KEYWORDS = [
        "panic", "overwhelmed", "can't breathe", "too much", "freaking out",
        "shaking", "terrified", "dizzy", "spiraling", "suffocating",
        "extremely sad", "extremely angry", "extremely scared",
        "very sad", "very lost", "very angry"
    ]

    if any(word in user_message.lower() for word in GROUNDING_KEYWORDS):
        return jsonify({
            "reply": "It sounds like you're feeling overwhelmed. Let’s take a moment to breathe 💙",
            "emotion": "fear",
            "exit_intent": False,
            "celebration_type": None,
            "trigger_grounding": True
        })

    exit_intent = is_exit_intent(user_message)

    try:
        if not session.get("chat_started"):
            session["chat_started"] = True
            reply, meta = first_message(user_message)
        elif exit_intent:
            return jsonify({
                "reply": "I'm really glad we talked today. Thank you for visiting <strong>Mann Mitra</strong> 💙",
                "emotion": "neutral",
                "exit_intent": True,
                "celebration_type": None
            })
        else:
            reply, meta = continue_convo(user_message)

        return jsonify({
            "reply": reply,
            "emotion": meta.get("emotion", "neutral"),
            "exit_intent": exit_intent,
            "celebration_type": meta.get("celebration_type", None)
        })

    except Exception as e:
        print("💥 Error:", e)
        return jsonify({
            "reply": "Oops, I’m having trouble replying right now. Please try again later.",
            "emotion": "neutral",
            "exit_intent": False,
            "celebration_type": None
        })

@app.route("/clear_session", methods=["POST"])
def clear_session():
    session.clear()
    resp = make_response("", 204)
    resp.set_cookie("session", "", expires=0)
    return resp



