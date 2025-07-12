import os
import random
import json
import re
from dotenv import load_dotenv
from langdetect import detect
import google.generativeai as genai
from transformers import pipeline


# Load and configure Gemini API key securely
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Gemini model with memory
aasha_session = genai.GenerativeModel("models/gemini-2.5-flash").start_chat(history=[])

# Translation pipeline (Hindi → English)
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en")
def translate_to_english(text):
    return translator(text)[0]['translation_text']

# Emotion classification using GoEmotions
emotion_classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=5)

# GoEmotions → Aasha categories
GOEMOTION_TO_CORE = {
    "admiration": "love", "amusement": "joy", "approval": "love", "caring": "love",
    "desire": "love", "excitement": "joy", "gratitude": "love", "joy": "joy",
    "love": "love", "optimism": "joy", "pride": "joy", "relief": "joy", "surprise": "surprise",
    "anger": "anger", "annoyance": "anger", "disapproval": "anger", "disgust": "anger",
    "embarrassment": "sadness", "fear": "fear", "nervousness": "fear", "confusion": "fear",
    "sadness": "sadness", "grief": "sadness", "remorse": "sadness",
    "realization": "neutral", "curiosity": "neutral", "neutral": "neutral"
}


# Emotion-based reflections and ideas
emotion_responses = {
    "sadness": {"reflection": "That sounds incredibly heavy — I’m really sorry you're carrying this.",
                 "ideas": ["Wrap up in a soft blanket and sip something warm", "Try writing what you’re feeling, even messily", "Listen to a soft, comforting song"]},
    "fear": {"reflection": "It’s completely okay to feel scared — you’re not alone in this.",
              "ideas": ["Try naming five things around you to ground yourself", "Take a few slow belly breaths", "Hold onto something soft and familiar"]},
    "anger": {"reflection": "That kind of anger can feel overwhelming — and it’s valid.",
               "ideas": ["Scribble or draw your emotions without judgment", "Write down what you wish you could say", "Move around — shake out your arms or take a brisk walk"]},
    "joy": {"reflection": "That’s so lovely to hear — I’m smiling with you.",
             "ideas": ["Close your eyes and really soak it in", "Capture it in a photo or note to remember", "Share it with someone who cares"]},
    "love": {"reflection": "That warm feeling is so special — thank you for sharing it.",
              "ideas": ["Text someone what they mean to you", "Write down how that love feels", "Breathe deeply and just hold onto the moment"]},
    "surprise": {"reflection": "That must’ve caught you off guard — surprises stir up so much.",
                 "ideas": ["Pause and take a slow breath", "Note your first thoughts about what happened", "Just sit quietly and let it settle"]},
    "neutral": {"reflection": "Whatever you're feeling, I'm right here with you.",
                "ideas": ["Take a short pause — maybe a breath or gentle stretch", "Write down anything on your mind", "Put on some soft background music"]}
}

# Load FAQ database
with open("faq.json", "r") as f:
    faq_data = json.load(f)

def match_faq(user_input):
    clean = user_input.lower().strip()
    for entry in faq_data:
        for q in entry["questions"]:
            if q in clean:
                return entry["answer"]
    return None

# Celebration type classifier
def detect_celebration_type(message):
    msg = message.lower()
    if any(k in msg for k in ["anniversary", "years together", "special day"]): return "hearts"
    if "job" in message and any(word in message for word in ["got", "hired", "new", "landed"]): return "confetti"
    if any(k in msg for k in ["birthday", "bday"]): return "balloons"
    return None

INVITE_LINES = [
    "I'm here if you want to share more.",
    "Whenever you feel ready to talk more, I'm listening.",
    "Feel free to open up as much or as little as you’d like.",
    "Only if you want — I’m always here.",
]

SHORT_REACT = {
    "joy": ["That sounds wonderful!", "That’s such good energy!", "So glad you’re feeling this!"],
    "love": ["That warmth really comes through.", "What a lovely moment to hold onto.", "That's such a tender feeling."],
    "surprise": ["That must’ve caught you off guard!", "Wow, I wasn’t expecting that either.", "Life’s full of little twists, isn’t it?"]
}

def get_emotion_label(text):
    try:
        # 🌍 Detect language
        lang = detect(text)
        print(f"🌍 Detected language: {lang}")
        print("📝 Text:", text)

        # Translate if needed
        word_count = len(re.findall(r'\w+', text))
        if lang != "en" and word_count > 3:
            print("🌐 Translating to English...")
            text = translate_to_english(text)
        else:
            print("✅ Using original text")

        # 🔁 Early keyword override
        text_lower = text.lower()
        emotion_keywords = {
            "sadness": ["sad", "grief", "loss", "hopeless", "down", "depressed", "cry"],
            "joy": ["happy", "excited", "yay", "glad", "smile", "fun", "joy", "cheerful"],
            "anger": ["angry", "mad", "furious", "rage", "irritated", "annoyed"],
            "fear": ["anxious", "worried", "scared", "afraid", "panic", "nervous", "terrified"],
            "love": ["love", "loved", "loving", "dear", "affection", "caring", "heartfelt"],
            "surprise": ["shocked", "surprised", "wow", "unexpected", "can't believe"],
            "neutral": ["okay", "fine", "meh", "nothing", "normal"]
        }

        for core_emotion, keywords in emotion_keywords.items():
            if any(word in text_lower for word in keywords):
                print(f"🔁 Strong override to '{core_emotion}' based on keyword match")
                return core_emotion

        # 🎯 Get emotion predictions from GoEmotions
        results = emotion_classifier(text)[0]

        # 🔄 Convert GoEmotions → Core 7
        core_scores = {}
        for r in results:
            label = r["label"].lower()
            score = r["score"]
            core = GOEMOTION_TO_CORE.get(label, "neutral")
            core_scores[core] = core_scores.get(core, 0) + score

        # 🎯 Pick top core emotion
        top_emotion = max(core_scores.items(), key=lambda x: x[1])[0]

        # 💖 Override to "love" if joyful text contains love-related words
        if top_emotion in ["joy", "neutral"] and any(word in text_lower for word in emotion_keywords["love"]):
            print("💖 Overriding emotion to 'love' due to love-related words in joyful context")
            top_emotion = "love"

        print("🧠 Emotion detected:", top_emotion)
        return top_emotion

    except Exception as e:
        print("❌ Emotion error:", e)
        return "neutral"

# First response

def first_message(user_input):
    faq = match_faq(user_input)
    if faq:
        return faq, {"emotion": "neutral", "celebration_type":None}

    emotion = get_emotion_label(user_input)
    celebration = detect_celebration_type(user_input)

    # For light emotions → keep it brief and bright
    if emotion in ["joy", "love", "surprise"]:
        reaction = random.choice(SHORT_REACT[emotion])
        suggestion = random.choice(emotion_responses[emotion]["ideas"])
        invite = random.choice(INVITE_LINES)
        response = f"{reaction} {suggestion}. {invite}"
        return response, {"emotion": emotion, "celebration_type": celebration}
    
    # For heavier emotions → use Gemini with detailed structure
    prompt = f'''
You are Aasha, a deeply emotionally intelligent AI companion. 
Speak with warmth, empathy, and clarity — like a close, thoughtful friend.

This is the user's first message:
"{user_input}"

Please:
- Start with a short emotional reflection (2 lines max)
- Offer 2 gentle, supportive ideas based on their emotion
- End with a soft invitation to share more, if they’d like
- Keep the tone human, warm, not robotic
- Never use endearments like "dear" or "sweetheart"
- If the message is vague or low-detail, be brief
'''

    try:
        response = aasha_session.send_message(prompt)
        return response.text.strip(), {"emotion": emotion, "celebration_type": celebration}
    except Exception as e:
        print("Gemini error:", e)
        return "I'm here with you, but something's a little off on my side. Want to try again?"

# Ongoing conversation

def continue_convo(user_input):
    faq = match_faq(user_input)
    if faq:
        return faq, {"emotion": "neutral", "celebration_type": None}

    emotion = get_emotion_label(user_input)
    celebration = detect_celebration_type(user_input)

    # For light emotions, skip Gemini to keep it snappy
    if emotion in ["joy", "love", "surprise"]:
        reflection = random.choice(SHORT_REACT[emotion])
        suggestion = random.choice(emotion_responses[emotion]["ideas"])
        invite = random.choice(INVITE_LINES)
        response = f"{reflection} {suggestion}. {invite}"
        return response, {"emotion": emotion, "celebration_type": celebration}

    # For heavier or neutral feelings → use Gemini
    prompt = f'''
You are Aasha — an emotionally intelligent AI companion who remembers past conversations and emotions.
Your tone is warm, clear, and comforting — like a close friend who truly listens.

Here’s the user’s message:
"{user_input}"

Please:
- Respond in 3 to 4 short, natural sentences.
- Acknowledge what they’re feeling now (even if it's vague or mixed).
- If relevant, gently refer to earlier tone/emotion without sounding scripted.
- Offer 1 or 2 soft, grounded suggestions or reflections.
- End with a varied, warm invitation to keep talking (see examples below).
- Avoid clinical language, repetitive advice, or endearments.

Example invitations:
"I’m here if you want to share more."
"Only if you feel like it — I’m listening."
"Let me know if anything else is on your mind."
"Want to talk more about that?"
"Totally okay if not, but I'm here."
'''

    try:
        response = aasha_session.send_message(prompt)
        return response.text.strip(), {"emotion": emotion, "celebration_type": celebration}
    except Exception as e:
        print("Gemini error:", e)
        return "Hmm, something got tangled in my thoughts. Can we try that again?"

# Exit detection (optional intent classifier)
intent_classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion", top_k=1)

def is_exit_intent(text):
    try:
        lowered = text.lower()
        exit_phrases = ["bye", "goodbye", "see you", "talk to you later", "exit", "quit", "thanks, that’s all", "i have to go", "okay bye", "cya", "ttyl", "done chatting"]
        if any(p in lowered for p in exit_phrases):
            return True

        res = intent_classifier(text)
        label = res[0]['label'].lower()
        return "gratitude" in label or "goodbye" in label
    except Exception as e:
        print("Exit intent error:", e)
        return False

# CLI test mode
if __name__ == "__main__":
    print("Hi, I’m Aasha. What’s on your mind today?")
    user_input = input("You: ")
    print("Aasha:", first_message(user_input)[0])

    while True:
        user_input = input("You: ")
        if is_exit_intent(user_input):
            print("Aasha: I'm really glad we talked today. Please take care 💙")
            break
        print("Aasha:", continue_convo(user_input)[0])
