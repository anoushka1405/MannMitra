from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import google.generativeai as genai
from Aasha_chatbot import get_emotion_label, build_aasha_prompt, get_reply_with_memory

# Load env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
print("🔑 Google API Key loaded is:", GOOGLE_API_KEY)

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

app = Flask(__name__)

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
    try:
        reply, emotion = get_reply_with_memory(user_message)
        return jsonify({
            "reply": reply,
            "emotion": emotion
        })
    except Exception as e:
        print("Chat Error:", e)
        return jsonify({
            "reply": "Oops, I’m having trouble replying right now. Please try again later.",
            "emotion": "neutral"
        })

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5050)


