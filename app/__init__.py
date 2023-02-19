import re
import string
import os
import sys
from flask import Flask, request
from stop_words import get_stop_words

sys.path.append("./")
from modules.bayes import predict

# Get env port and host
PORT = os.getenv("PORT") or 8080
HOST = "localhost"

app = Flask(__name__)
symbol_pattern = rf'[{string.punctuation}]'
stop_words = get_stop_words("indonesian")

@app.route("/")
def main():
    text = request.args.get('text')

    if text:
        # Stemming
        text = re.sub(symbol_pattern, "", text)
        words = text.split(" ")

        # Stop words filter
        for word in words:
            if stop_words.__contains__(word):
                words.remove(word)
        return {
            "class": predict(" ".join(words)),
            "text": text,
            "clean_text": " ".join(words)
        }, 200
    else:
        return "Bad request", 400

if __name__ == "__main__":
    app.run(host=HOST, port=PORT)