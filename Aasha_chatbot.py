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

# Try multiple translation models for better accuracy
try:
    # Primary translator
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en")
    # Backup translator for better quality
    translator_backup = pipeline("translation", model="facebook/nllb-200-distilled-600M", 
                                src_lang="hin_Deva", tgt_lang="eng_Latn")
except:
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en")
    translator_backup = None


def translate_to_english(text):
    try:
        # Try primary translator first
        result = translator(text)[0]['translation_text']
        
        # If result looks poor (very short or contains weird artifacts), try backup
        if translator_backup and (len(result) < len(text) * 0.5 or any(char in result for char in ['<', '>', '@'])):
            backup_result = translator_backup(text)[0]['translation_text']
            if len(backup_result) > len(result):
                result = backup_result
        
        return result
    except:
        return text


# Emotion classification using GoEmotions
emotion_classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=5)

# GoEmotions → Aasha categories
GOEMOTION_TO_CORE = {
    "admiration": "love", "amusement": "joy", "approval": "neutral", "caring": "love",
    "desire": "love", "excitement": "joy", "gratitude": "love", "joy": "joy",
    "love": "love", "optimism": "joy", "pride": "joy", "relief": "joy", "surprise": "surprise",
    "anger": "anger", "annoyance": "anger", "disapproval": "anger", "disgust": "anger",
    "embarrassment": "sadness", "fear": "fear", "nervousness": "fear", "confusion": "fear",
    "sadness": "sadness", "grief": "sadness", "remorse": "sadness", "disappointment": "sadness",
    "realization": "neutral", "curiosity": "neutral", "neutral": "neutral"
}

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


# Expanded Hindi emotion keywords with more variations
EXPANDED_HINDI_KEYWORDS = {
    "anger": [
        # Direct anger words
        "gussa", "naraz", "gusse", "khafa", "bura laga", "bura lag", 
        "chillaya", "chillana", "jhagda", "bhadak", "taana", "tana", 
        "bardasht", "bardaasht", "chod do", "chod de", "chhod do",
        "pareshan", "tang", "khoon", "dimag", "sir", "pagal",
        "kitna jhelun", "hadd hoti hai", "punch a wall",
        
        # Contextual anger phrases
        "kya bakwas", "bakwas hai", "faltu", "bewakoof", "stupid",
        "kya yaar", "yaar kya", "khatam kar", "bas kar", "band kar",
        "dikkat", "problem", "musibat", "takleef", "귀찮",
        
        # Family/social anger
        "waalon ne", "walon ne", "taana mara", "taana maara", "sunaya",
        "bolte hain", "kehte hain", "bola", "kaha", "comment",
        
        # Intensity words that indicate anger
        "phir se", "fir se", "har bar", "har baar", "hamesha", "again"
    ],
    
    "fear": [
        # Direct fear words
        "dar", "darr", "ghabrahat", "ghabra", "bechain", "bechaini",
        "asurakshit", "unsafe", "dil tez", "dil ki", "sankat",
        "sahas nahi", "himmat nahi", "kharab", "bura", "galat",
        "dar lag raha hai", "galat na ho jaye", "har waqt dar", "darr laga rehta hai",
        
        # Anxiety and worry
        "tension", "stress", "pareshani", "chinta", "fikar", "worry",
        "nervous", "scared", "afraid", "panic", "anxious",
        
        # Contextual fear phrases
        "kya hoga", "kya ho", "ho jaye", "ho gaya", "ho jayega",
        "lag raha", "lagta hai", "laga", "feel", "feeling",
        "kuch galat", "kuch bura", "something wrong", "something bad",
        
        # Physical symptoms of fear
        "saas", "breath", "dil", "heart", "kaamp", "shiver", "thrill"
    ],
    
    "sadness": [
        # Direct sadness words
        "dukh", "udaas", "udas", "toot", "toota", "ro", "rona", "rone",
        "niraash", "nirash", "disappointed", "thak", "tired", "thaka",
        "haar", "har", "akela", "alone", "khaali", "empty", "dard",
        "khamoshi", "kaafi hai ab", "nothing ever changes", "what's the point",
        
        # Hopelessness
        "bekaar", "bekar", "useless", "waste", "kya fayda", "koi fayda",
        "kuch nahi", "nothing", "sab kuch", "everything", "zindagi",
        "life", "jeena", "living", "jee", "death", "maut", "mar",
        
        # Emotional pain
        "dil", "heart", "aansu", "tears", "cry", "weep", "sob",
        "heavy", "bhaari", "burden", "load", "weight", "bojh",
        
        # Regret and remorse
        "regret", "pachtawa", "galti", "mistake", "should have", "kash",
        "kaash", "wish", "agar", "if only", "why", "kyu", "kyun"
    ],
    
    "joy": [
        # Direct joy words
        "khush", "khushi", "khushii", "happy", "mast", "mazedar",
        "maja", "maza", "maje", "fun", "enjoy", "smile", "has",
        "hansi", "laugh", "sukoon", "peace", "shanti", "calm",
        "bahut maza aaya", "sach mein", "doston ke saath ghumna",
        
        # Excitement
        "excited", "josh", "energy", "awesome", "amazing", "wow",
        "great", "accha", "acha", "achha", "good", "best", "badiya",
        "badhiya", "zabardast", "kamaal", "wonderful", "fantastic",
        
        # Achievement and success
        "success", "safalta", "kamyabi", "jeet", "jeeta", "win", "won",
        "pass", "passed", "clear", "achieve", "mil gaya", "mil gayi",
        
        # Celebration
        "party", "celebrate", "manana", "khushi manana", "festival",
        "birthday", "anniversary", "special", "khas", "moment"
    ],
    
    "love": [
        # Direct love words
        "pyar", "pyaar", "mohabbat", "love", "loved", "loving",
        "apnapan", "apna", "apne", "dil se", "heartfelt", "heart",
        "sambhala", "sambhalna", "khayal", "care", "caring", "cared",
        "zindagi adhoori", "made my whole day",
        
        # Affection and closeness
        "lagav", "attachment", "bond", "rishta", "relation", "close",
        "najdeek", "paas", "saath", "together", "tumse", "tujhse",
        "aapse", "you", "yaar", "friend", "dost", "buddy", "bhai",
        
        # Gratitude and appreciation
        "thank", "thanks", "dhanyawad", "shukriya", "grateful",
        "appreciate", "value", "keemat", "important", "zaroori",
        "matter", "matlab", "mean", "special", "khas", "dear"
    ],
    
    "surprise": [
        # Direct surprise words
        "shocked", "surprise", "surprised", "hairan", "heran",
        "ajeeb", "ajib", "strange", "unexpected", "achanak",
        "suddenly", "sudden", "kaise", "how", "kya", "what",
        "what just happened", "socha hi nahi tha", "wait what happened",
        
        # Disbelief
        "believe nahi", "nahi believe", "can't believe", "yakeen nahi",
        "bharosa nahi", "impossible", "asambhav", "ho nahi sakta",
        "nahi ho sakta", "really", "sach", "sachii", "truly",
        
        # Exclamations
        "wow", "omg", "oh my god", "are", "arre", "kya baat",
        "kamal", "amazing", "incredible", "unbelievable", "mind blown",
        "dimag", "not expected", "expect nahi", "socha nahi", "think nahi"
    ]
}

# Improved keyword detection with phrase matching
def detect_emotion_keywords(text, lang="en"):
    text_lower = text.lower()
    
    # For Hindi, use expanded keywords
    if lang == "hi":
        keywords = EXPANDED_HINDI_KEYWORDS
    else:
        keywords = {
            "sadness": ["sad", "grief", "loss", "hopeless", "down", "depressed", "cry", "alone", "tired", "numb", "hurt", "lonely", "broken", "regret", "disappointed", "devastated"],
            "joy": ["happy", "excited", "yay", "glad", "smile", "fun", "cheerful", "bright", "laugh", "peaceful", "grateful", "thrilled", "delighted", "wonderful"],
            "anger": ["angry", "mad", "furious", "rage", "irritated", "annoyed", "hate", "frustrated", "pissed", "bitter", "livid", "outraged", "disgusted"],
            "fear": ["anxious", "worried", "scared", "afraid", "panic", "nervous", "terrified", "unsafe", "shaking", "tension", "frightened", "alarmed"],
            "love": ["love", "loved", "cared", "affection", "heartfelt", "close to me", "dear", "bond", "sweet", "caring", "adore", "cherish"],
            "surprise": ["shocked", "surprised", "unexpected", "can't believe", "wow", "unbelievable", "sudden", "mind blown", "astonished", "amazed"],
            "neutral": ["okay", "fine", "meh", "nothing", "normal", "usual", "bored", "whatever", "idk", "alright"]
        }
    
    # Score each emotion based on keyword matches
    emotion_scores = {}
    for emotion, emotion_keywords in keywords.items():
        score = 0
        for keyword in emotion_keywords:
            # Use regex for word boundaries to avoid partial matches
            if re.search(rf"\b{re.escape(keyword)}\b", text_lower):
                score += 1
            # Also check for phrase matches (for multi-word expressions)
            elif keyword in text_lower:
                score += 0.5
        emotion_scores[emotion] = score
    
    # Return the emotion with highest score if any keywords matched
    if any(score > 0 for score in emotion_scores.values()):
        return max(emotion_scores.items(), key=lambda x: x[1])[0]
    
    return None


def get_emotion_label(text):
    try:
        text_lower = text.lower()
        original_text = text

        # Step 1: Enhanced Language Detection
        HINDI_HINTS = [
            "mujhe", "dukh", "hoon", "udaas", "kyu", "kyun", "aisa", "nahi", 
            "zindagi", "ro", "toot", "kya", "kaise", "khushi", "gaya", "hai",
            "mein", "me", "kar", "se", "ko", "ki", "ka", "ke", "wala", "wali"
        ]
        
        # Count Hindi hints for better detection
        hindi_count = sum(1 for hint in HINDI_HINTS if re.search(rf"\b{re.escape(hint)}\b", text_lower))
        lang = "hi" if hindi_count >= 2 else detect(text)

        # Step 2: Enhanced Keyword Detection (run first, before translation)
        keyword_emotion = detect_emotion_keywords(text, lang)
        if keyword_emotion:
            print(f"🎯 Keyword override: {keyword_emotion}")
            return keyword_emotion

        # Step 3: Special override phrases (expanded)
        success_phrases = [
            "but i passed", "but it worked", "but it turned out okay",
            "lekin pass", "lekin ho gaya", "par ho gaya", "but mil gaya"
        ]
        if any(phrase in text_lower for phrase in success_phrases):
            print("🎯 Override: success after challenge → joy")
            return "joy"

        # Step 4: Translate if Hindi and longer than 2 words
        word_count = len(re.findall(r'\w+', text))
        if lang == "hi" and word_count > 2:
            print("🌐 Translating to English...")
            text = translate_to_english(text)
            print(f"📝 Translation: '{original_text}' → '{text}'")
            
            # Run keyword detection again on translated text
            keyword_emotion = detect_emotion_keywords(text, "en")
            if keyword_emotion:
                print(f"🎯 Post-translation keyword override: {keyword_emotion}")
                return keyword_emotion
        else:
            print("✅ Using original text")

        # Step 5: Emotion Prediction (GoEmotions)
        results = emotion_classifier(text)
        if isinstance(results, list) and len(results) > 0:
            results = results[0] if isinstance(results[0], list) else results
        
        print("🔍 GoEmotions raw:", results)

        # Step 6: Convert to Core 7 Emotion Categories with better scoring
        core_scores = {}
        for r in results:
            label = r["label"].lower()
            score = r["score"]
            core = GOEMOTION_TO_CORE.get(label, "neutral")
            core_scores[core] = core_scores.get(core, 0) + score

        # Step 7: Adjust confidence threshold and add context-aware overrides
        top_emotion, top_score = max(core_scores.items(), key=lambda x: x[1])
        
        # Lower confidence threshold for translated Hindi
        confidence_threshold = 0.2 if lang == "hi" else 0.3
        
        if top_score < confidence_threshold:
            print(f"🛑 Low confidence ({top_score:.2f}) — checking context...")
            
            # Context-based fallbacks for low confidence
            if any(word in text_lower for word in ["kya", "kaise", "how", "what", "why", "kyu"]):
                print("🔄 Question context detected → neutral")
                return "neutral"
            elif any(word in text_lower for word in ["dar", "afraid", "scared", "nervous"]):
                print("🔄 Fear context detected → fear")
                return "fear"
            else:
                print("🔄 Defaulting to neutral")
                return "neutral"

        # Step 8: Final context overrides
        if top_emotion in ["joy", "neutral"]:
            love_indicators = ["love", "care", "close", "friend", "pyar", "dost", "apna"]
            if any(word in text_lower for word in love_indicators):
                print("💖 Context override → love")
                return "love"

        print(f"🧠 Final emotion: {top_emotion} (confidence: {top_score:.2f})")
        return top_emotion

    except Exception as e:
        print("❌ Emotion detection error:", e)
        return "neutral"
    
SHORT_REACT = {
    "joy": [
        "That’s such good energy! I’m smiling with you.",
        "That kind of joy deserves to be felt fully — I’m so glad you shared it with me.",
        "A moment like this? Worth holding onto. I’m right here with you."
    ],
    "love": [
        "That warmth really comes through.",
        "What a lovely moment to hold onto.",
        "That's such a tender feeling."
    ],
    "surprise": [
        "That must’ve caught you off guard!",
        "Wow, I wasn’t expecting that either.",
        "Life’s full of little twists, isn’t it?"
    ]
}

JOY_INVITES = [
    "Want to tell me what made your day feel this good?",
    "If you feel like sharing what’s lighting you up — I’d love to hear.",
    "What’s been bringing this kind of smile today?",
    "What’s been going *right* for you lately?"
]

INVITE_LINES = [
    "If you want to say more, I’m right here.",
    "Want to talk a little more about this?",
    "I’d love to hear more if you’re open to sharing.",
    "If there’s more on your mind, I’m listening."
]

# ---------------------- First Message Handler ----------------------

def first_message(user_input, lang=None):
    # Step 1: Language Detection
    try:
        lang = detect(user_input)
    except:
        lang = "en"

    # Step 2: FAQ Matching
    faq_response = match_faq(user_input)
    if faq_response:
        return faq_response, {"emotion": "neutral", "celebration_type": None}

    # Step 3: Emotion & Celebration Detection
    emotion = get_emotion_label(user_input)
    celebration = detect_celebration_type(user_input)
    word_count = len(re.findall(r'\w+', user_input))

    # Step 4: Short Reactions for Light Messages
    if emotion in ["joy", "love", "surprise"] and word_count < 12:
        reaction = random.choice(SHORT_REACT[emotion])
        suggestion = random.choice(emotion_responses[emotion]["ideas"])
        invite = random.choice(JOY_INVITES if emotion == "joy" else INVITE_LINES)
        response = f"{reaction} {suggestion}. {invite}"
        return response, {"emotion": emotion, "celebration_type": celebration}

    # Step 5: Gemini Prompt for Heavier or Longer Messages
    if lang == "hi":
        prompt = f'''
आप आशा हो — एक संवेदनशील और समझदार AI साथी।

यह उपयोगकर्ता का पहला संदेश है:
"{user_input}"

कृपया:
- एक छोटी सी भावनात्मक प्रतिक्रिया दें (2 लाइन से कम)
- उनके जज़्बातों के आधार पर 2 हल्के सुझाव दें
- अंत में एक सॉफ्ट आमंत्रण दें कि वे चाहें तो और बात कर सकते हैं
- स्वर इंसानी और गर्मजोशी भरा होना चाहिए — रोबोटिक नहीं
- कृपया "डियर" या "प्रिय" जैसे संबोधन न करें
- अगर संदेश अस्पष्ट या छोटा है, तो जवाब भी छोटा रखें
'''
    else:
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

    # Step 6: Gemini Response
    try:
        response = aasha_session.send_message(prompt)
        return response.text.strip(), {"emotion": emotion, "celebration_type": celebration}
    except Exception as e:
        print("Gemini error:", e)
        fallback_message = "I'm here with you, but something's a little off on my side. Want to try again?"
        return fallback_message, {"emotion": "neutral", "celebration_type": None}


# ---------------------- Conversation Continuation ----------------------

def continue_convo(user_input, lang=None):
    faq = match_faq(user_input)
    if faq:
        return faq, {"emotion": "neutral", "celebration_type": None}

    emotion = get_emotion_label(user_input)

    try:
        lang = detect(user_input)
    except:
        lang = "en"


    # If celebration triggers (TODO: move to detect_celebration_type())
    msg = user_input.lower()
    if any(k in msg for k in ["anniversary", "years together", "special day"]): return "hearts"
    if "job" in msg and any(word in msg for word in ["got", "hired", "new", "landed"]): return "confetti"
    if any(k in msg for k in ["birthday", "bday"]): return "balloons"
    if any(k in msg for k in ["graduated", "degree", "passed"]): return "stars"
    if "baby" in msg or "welcomed our baby" in msg: return "stars"
    if "independence" in msg or "🇮🇳" in msg: return "flag"
    if "diwali" in msg or "lights everywhere" in msg: return "lights"
    if "party" in msg: return "confetti"

    # Gemini prompt
    if lang == "hi":
        prompt = f'''
आप आशा हो — एक संवेदनशील AI साथी जो पहले की बातचीत और भावनाओं को याद रखती है।
आपका लहजा गर्मजोशी भरा, साफ़, और सहायक होना चाहिए — जैसे एक करीबी दोस्त।

यह उपयोगकर्ता का संदेश है:
"{user_input}"

कृपया:
- 3-4 छोटी, स्वाभाविक पंक्तियों में जवाब दें
- जो वे अभी महसूस कर रहे हैं, उसे पहचानें (चाहे वह अस्पष्ट ही क्यों न हो)
- यदि प्रासंगिक हो, तो पहले के लहजे/भाव का हल्का सा संदर्भ दें
- 1-2 कोमल, ग्राउंडेड सुझाव दें
- अंत में एक सहज आमंत्रण दें जैसे:
  "अगर और बात करना चाहें तो मैं यहीं हूँ।"
  "सिर्फ़ अगर आपका मन हो — मैं सुन रही हूँ।"
  "कुछ और कहना चाहें तो बेहिचक बताएं।"
'''
    else:
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
'''

    try:
        response = aasha_session.send_message(prompt)
        return response.text.strip(), {"emotion": emotion, "celebration_type": celebration}
    except Exception as e:
        print("Gemini error:", e)

    return "I'm here with you, but something's a little off on my side. Want to try again.", {
        "emotion": "neutral",
        "celebration_type": None
    }

# ---------------------- Exit Intent Detection ----------------------

intent_classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion", top_k=1)

def is_exit_intent(text):
    try:
        lowered = text.lower()
        exit_phrases = [
            # English exits
            "bye", "goodbye", "see you", "talk to you later", "catch you later", "ttyl",
            "gotta go", "i have to go", "logging off", "i’m done", "signing off", "good night",
            # Hindi exits
            "milte hain", "phir milenge", "ab chalta hoon", "ab chalti hoon", "shubh ratri",
            "kal milte hain", "baad mein baat", "accha chalta hoon", "chalo bye", "alvida",
            # Mixed
            "ok bye", "bye aasha", "kal milte", "itna hi", "milte hai baad mein",
            "i'm leaving now", "main chalta hoon", "enough for now", "goodnight"
        ]

        if any(p in lowered for p in exit_phrases):
            return True

        res = intent_classifier(text)
        while isinstance(res, list) and len(res) > 0:
            res = res[0]

        if isinstance(res, dict) and "label" in res:
            label = res["label"].lower()
            return "gratitude" in label or "goodbye" in label

    except Exception as e:
        print("Exit intent error:", e)

    return False

# ---------------------- Command Line Interface ----------------------

if __name__ == "__main__":
    print("Hi, I’m Aasha. What’s on your mind today?")
    user_input = input("You: ")
    response, meta = first_message(user_input)
    print("Aasha:", response)

    while True:
        user_input = input("You: ")
        if is_exit_intent(user_input):
            print("Aasha: I'm really glad we talked today. Please take care 💙")
            break

        response, meta = continue_convo(user_input)
        print("Aasha:", response)
