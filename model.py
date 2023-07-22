import logging
from flask import Flask, render_template, request, jsonify
from transformers import pipeline

# Configure logging to show model download progress
logging.basicConfig(level=logging.INFO)

app = Flask(__name__, template_folder='clientside', static_folder='clientside')

def text_summarization(text, model_name="sshleifer/distilbart-cnn-12-6", max_length=150, min_length=30):
    summarizer = pipeline("summarization", model=model_name, tokenizer=model_name)
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=True)
    return summary[0]['summary_text']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if request.headers['Content-Type'] == 'application/json':
        data = request.json
    else:
        data = request.form

    paragraph = data['text']
    word = data.get('keyword')

    # Preprocess the input text
    if word:
        input_text = f"Summarize the paragraph containing the word '{word}': {paragraph}"
    else:
        input_text = f"Summarize the paragraph: {paragraph}"

    summary = text_summarization(input_text)

    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)