import google.generativeai as genai
from transformers import pipeline
import random

# Configure Gemini
genai.configure(api_key="YOUR_API_KEY")  # Replace with your real key

# Gemini model with memory
model = genai.GenerativeModel("models/gemini-2.5-flash")
aasha_session = model.start_chat(history=[])

# Emotion detection pipeline
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

# Emotion-specific content
emotion_responses = {
    "sadness": {
        "reflection": "That sounds incredibly heavy — I’m really sorry you're carrying this.",
        "ideas": [
            "Wrap up in a soft blanket and sip something warm",
            "Try writing what you’re feeling, even messily",
            "Listen to a soft, comforting song"
        ]
    },
    "fear": {
        "reflection": "It’s completely okay to feel scared — you’re not alone in this.",
        "ideas": [
            "Try naming five things around you to ground yourself",
            "Take a few slow belly breaths",
            "Hold onto something soft and familiar"
        ]
    },
    "anger": {
        "reflection": "That kind of anger can feel overwhelming — and it’s valid.",
        "ideas": [
            "Scribble or draw your emotions without judgment",
            "Write down what you wish you could say",
            "Move around — shake out your arms or take a brisk walk"
        ]
    },
    "joy": {
        "reflection": "That’s so lovely to hear — I’m smiling with you.",
        "ideas": [
            "Close your eyes and really soak it in",
            "Capture it in a photo or note to remember",
            "Share it with someone who cares"
        ]
    },
    "love": {
        "reflection": "That warm feeling is so special — thank you for sharing it.",
        "ideas": [
            "Text someone what they mean to you",
            "Write down how that love feels",
            "Breathe deeply and just hold onto the moment"
        ]
    },
    "surprise": {
        "reflection": "That must’ve caught you off guard — surprises stir up so much.",
        "ideas": [
            "Pause and take a slow breath",
            "Note your first thoughts about what happened",
            "Just sit quietly and let it settle"
        ]
    },
    "neutral": {
        "reflection": "Whatever you're feeling, I'm right here with you.",
        "ideas": [
            "Take a short pause — maybe a breath or gentle stretch",
            "Write down anything on your mind",
            "Put on some soft background music"
        ]
    }
}

# Emotion label detector
def get_emotion_label(text):
    try:
        result = emotion_classifier(text)

        # Check for both flat and nested list output
        if isinstance(result, list):
            # Case: [[{'label': 'joy'}]]
            if isinstance(result[0], list) and 'label' in result[0][0]:
                label = result[0][0]['label'].lower()
            # Case: [{'label': 'joy'}]
            elif isinstance(result[0], dict) and 'label' in result[0]:
                label = result[0]['label'].lower()
            else:
                label = "neutral"

            print("🧠 Emotion detected:", label)
            return label

    except Exception as e:
        print("Emotion detection error:", e)

    return "neutral"


# First interaction with Aasha
def first_message(user_input):
    emotion = get_emotion_label(user_input)
    response = emotion_responses.get(emotion, emotion_responses["neutral"])
    reflection = response["reflection"]
    suggestions = random.sample(response["ideas"], 2)

    intro_prompt = f"""
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

Example format:
It sounds like you’re carrying a lot right now. That’s totally okay.
Here are two ideas that might help:
– [idea 1]
– [idea 2]
If you feel like talking more, I’m here.
"""

    try:
        response = aasha_session.send_message(intro_prompt)
        return response.text.strip()
    except Exception as e:
        print("Gemini error in first_message:", e)
        return "I’m here with you, but I’m having a little trouble responding right now."

# Ongoing conversation with memory
def continue_convo(user_input):
    followup_prompt = f"""
You are Aasha — an emotionally intelligent AI companion who remembers past conversations and emotions.

Your tone is warm, clear, and comforting — like a close friend who truly listens. You do not use words like "sweetheart" or "dear".

Here’s the user’s message:
"{user_input}"

Please:
- Respond in 3 to 4 short, natural sentences.
- Acknowledge what they’re feeling now.
- Refer gently to what they shared earlier, *if relevant*.
- Offer 1 or 2 soft, specific ideas — emotional, creative, or grounding.
- End with a warm but non-pushy invitation to keep talking (“I’m here if you want to share more.”)
- Avoid clinical language or repeating ideas unless the user directly brings them up.
- If they express doubt or sadness, validate it, then gently guide.

Reply as Aasha only — no markdown, no formatting. Your voice is tender, calm, and human.
"""

    try:
        response = aasha_session.send_message(followup_prompt)
        return response.text.strip()
    except Exception as e:
        print("Gemini error in continue_convo:", e)
        return "Hmm, something got tangled in my thoughts. Can we try that again?"

# CLI test mode
if __name__ == "__main__":
    print("Hi, I’m Aasha. What’s on your mind today?")
    user_input = input("You: ")
    print("Aasha:", first_message(user_input))

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["bye", "exit", "quit"]:
            print("Aasha: I'm really glad we talked today. Please take care 💙")
            break
        print("Aasha:", continue_convo(user_input))
