# app.py

from flask import Flask, request, jsonify
import os
import json
import easyocr
from google.genai import Client
from google.genai.errors import APIError

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
# PROCESS RECEIPTS (OCR TEXT)
# ==========================================
def clean_receipt_with_gemini(raw_text):
    prompt = f"""
You are an assistant that extracts structured receipt data.

RULES:
- Sometimes a number appears BEFORE an item. This is quantity.
    Example: "2 BANANA" → item: "BANANA", quantity: 2
- If no quantity appears, assume quantity = 1.
- Extract ONLY item names, quantities, and unit prices.
- Ignore EVERYTHING else: TOTAL, CASH, CHANGE, store name, footer text, warnings.
- Keep item names clean and readable.
- Keep all prices exactly as seen.

OUTPUT STRICT JSON:
[
  {{"item": "ITEM_NAME", "quantity": NUMBER, "price": NUMBER}},
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

        # Strip ```json wrappers
        if text_output.startswith("```"):
            lines = text_output.split("\n")
            text_output = "\n".join(lines[1:-1]).strip()

        return json.loads(text_output)

    except Exception as e:
        print("Gemini processing error:", e)
        return []


# ==========================================
# PROCESS VOICE TEXT (SECOND AGENT)
# ==========================================
def parse_voice_text_with_gemini(text):
    prompt = f"""
You are an assistant that extracts structured shopping data from natural language text.

RULES:
- Text comes from voice transcription like:
  "Today I bought two bananas and a watermelon for 4 dollars"
- Extract ONLY purchased items, quantities, and prices.
- If quantity is mentioned → use it.
- If no quantity is mentioned → quantity = 1.
- If price is mentioned → use it.
- If no price mentioned → "price": null.
- Ignore everything that is not a purchased item.
- Ignore "today I bought", greetings, story parts, etc.

OUTPUT STRICT JSON:
[
  {{"item": "ITEM_NAME", "quantity": NUMBER, "price": NUMBER or null}},
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

        # Strip ```json wrappers
        if output.startswith("```"):
            lines = output.split("\n")
            output = "\n".join(lines[1:-1]).strip()

        return json.loads(output)

    except Exception as e:
        print("Error:", e)
        return []


# ==========================================
# OCR RECEIPT ENDPOINT
# ==========================================
@app.route("/parse_receipt", methods=["POST"])
def parse_receipt():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save file locally
    os.makedirs("ocr_images", exist_ok=True)
    image_path = os.path.join("ocr_images", file.filename)
    file.save(image_path)

    # OCR extraction
    ocr_results = reader.readtext(image_path)
    raw_text = "\n".join([t for (_, t, _) in ocr_results])

    # Clean with Gemini
    items = clean_receipt_with_gemini(raw_text)

    total = 0
    for item in items:
        total += float(item["price"]) * int(item["quantity"])

    return jsonify({
        "image": image_path,
        "raw_text": raw_text,
        "items": items,
        "total": round(total, 2)
    })


# ==========================================
# VOICE TEXT ENDPOINT (SECOND AGENT)
# ==========================================
@app.route("/parse_voice", methods=["POST"])
def parse_voice():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    user_text = data["text"]

    items = parse_voice_text_with_gemini(user_text)

    total = 0
    for item in items:
        if item["price"] is not None:
            total += item["price"] * item["quantity"]

    return jsonify({
        "raw_text": user_text,
        "items": items,
        "estimated_total": round(total, 2)
    })


# ==========================================
# ROOT
# ==========================================
@app.route("/")
def index():
    return "Finovia OCR API is running. POST to /parse_receipt or /parse_voice"


# ==========================================
# AUTO SWITCH PORT IF 5000 IS BUSY
# ==========================================
def run_server():
    import socket
    port = 5000
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("0.0.0.0", port)) != 0:
                break
            port += 1

    print(f"Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=True)


if __name__ == "__main__":
    run_server()
