from flask import Flask, request, jsonify
import os
from sentiment_analysis import analyze_sentiment, aggregate_results
os.environ["TRANSFORMERS_CACHE"] = "./transformerCache"
app = Flask(__name__)

@app.route("/")
def hello_world():
  """Responds with a hello message."""
  return "Hello from your Flask backend on Vercel!"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            transcript = f.read()

        # Perform sentiment analysis
        sentiment_results = analyze_sentiment(transcript)
        aggregated_results = aggregate_results(sentiment_results)

        response = {
            "chunk_results": sentiment_results,
            "aggregated_results": aggregated_results
        }

        return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
