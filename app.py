from flask import Flask, request, jsonify
import os
import json
import subprocess
import speech_recognition as sr
import easyocr
from google.genai import Client

app = Flask(__name__)

# ==========================================
# Gemini API
# ==========================================
API_KEY = "AIzaSyBAadRmBpy88DPtXjSLlx055iVds14NsWQ"
client = Client(api_key=API_KEY)

# ==========================================
# EasyOCR
# ==========================================
reader = easyocr.Reader(['en'])

# ==========================================
# AUDIO TRANSCRIPTION (Auto-convert any file to WAV)
# ==========================================
def transcribe_audio(file_path):
    wav_path = file_path + ".wav"

    # Convert ANY audio format → WAV (16kHz mono)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", file_path, "-ac", "1", "-ar", "16000", wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception as e:
        print("FFmpeg conversion failed:", e)
        return ""

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(wav_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return text
    except Exception as e:
        print("Transcription error:", e)
        return ""

# ==========================================
# GEMINI RECEIPT CLEANER
# ==========================================
def clean_receipt_with_gemini(raw_text):
    prompt = f"""
You are an assistant that extracts structured receipt data.

RULES:
- Numbers before an item = quantity.
- If no quantity → quantity = 1.
- Extract item name, quantity, unit (if any), and price.
- Ignore totals, cash, VAT, store name, etc.

OUTPUT STRICT JSON:
[
  {{"item": "ITEM", "quantity": NUMBER, "unit": "UNIT" or null, "price": NUMBER}},
  ...
]

Receipt text:
{raw_text}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        text_output = response.text.strip()

        if text_output.startswith("```"):
            lines = text_output.split("\n")
            text_output = "\n".join(lines[1:-1]).strip()

        return json.loads(text_output)

    except Exception as e:
        print("Gemini error:", e)
        return []

# ==========================================
# GEMINI VOICE PARSER
# ==========================================
def parse_voice_text_with_gemini(text):
    prompt = f"""
You extract purchased items from natural language voice text.

RULES:
- Example input: "I bought two bananas and a watermelon for 4 dollars"
- Extract ONLY items, quantities, unit, price.
- If quantity missing → quantity = 1
- If price missing → price = null
- If unit missing → unit = null

OUTPUT STRICT JSON:
[
  {{"item": "ITEM", "quantity": NUMBER, "unit": "UNIT" or null, "price": NUMBER or null}},
  ...
]

User text:
{text}
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        output = response.text.strip()

        if output.startswith("```"):
            lines = output.split("\n")
            output = "\n".join(lines[1:-1]).strip()

        return json.loads(output)

    except Exception as e:
        print("Gemini voice error:", e)
        return []

# ==========================================
# OCR ENDPOINT
# ==========================================
@app.route("/parse_receipt", methods=["POST"])
def parse_receipt():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    os.makedirs("ocr_images", exist_ok=True)
    image_path = os.path.join("ocr_images", file.filename)
    file.save(image_path)

    ocr_results = reader.readtext(image_path)
    raw_text = "\n".join([t for (_, t, _) in ocr_results])

    items = clean_receipt_with_gemini(raw_text)

    total = sum(
        float(item["price"]) * int(item["quantity"])
        for item in items if item["price"] is not None
    )

    return jsonify({
        "image": image_path,
        "raw_text": raw_text,
        "items": items,
        "total": round(total, 2)
    })

# ==========================================
# VOICE ENDPOINT (audio file → text → Gemini)
# ==========================================
@app.route("/parse_voice", methods=["POST"])
def parse_voice():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    os.makedirs("voice_uploads", exist_ok=True)
    audio_path = os.path.join("voice_uploads", file.filename)
    file.save(audio_path)

    # Convert + transcribe
    text = transcribe_audio(audio_path)
    if not text:
        return jsonify({"error": "Couldn't transcribe audio"}), 400

    items = parse_voice_text_with_gemini(text)

    total = sum(
        (item["price"] or 0) * item["quantity"]
        for item in items
    )

    return jsonify({
        "raw_text": text,
        "items": items,
        "estimated_total": round(total, 2)
    })

# ==========================================
# ROOT
# ==========================================
@app.route("/")
def index():
    return "Finovia OCR & Voice API is running."

# ==========================================
# PORT HANDLER
# ==========================================
def run_server():
    import socket
    port = 5000
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("0.0.0.0", port)) != 0:
                break
            port += 1

    print(f"Server running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)

if __name__ == "__main__":
    run_server()
