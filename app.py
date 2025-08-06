from flask import Flask, request, jsonify, render_template
from ollama_handler import generate_chat_response
from flask_cors import CORS

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    messages = data.get("messages")

    if messages and isinstance(messages, list) and len(messages) > 0:
        user_message = messages[-1]["content"]  # Get last user message
        response = generate_chat_response(user_message)  # Now uses classifier + RAG
        return jsonify(response)  # ✅ Return reply and source directly
    else:
        return jsonify({"reply": "⚠️ No input provided."}), 400


if __name__ == "__main__":
    app.run(debug=True)