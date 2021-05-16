from flask import Flask, redirect, url_for, render_template, request

from werkzeug.utils import secure_filename

import os

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random

import json
with open('intents.json') as file:
    data = json.load(file)



# This is this basic website code 
app = Flask(__name__)
@app.route('/')
def home():
    return 'Hello! This is where we can import your videos <h1>HELLO<h1>'

@app.route("/<name>")
def user(name):
           return f'hello{name}!'

@app.route("/admin")
def admin():
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run()

# up till this(up above is the basic website code).







app.config["IMAGE_UPLOADS"] = "/mnt/c/wsl/projects/pythonise/tutorials/flask_series/app/app/static/img/uploads"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]

def allowed_image(filename):

    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():

    if request.method == "POST":

        if request.files:

            image = request.files["image"]

            if image.filename == "":
                print("No filename")
                return redirect(request.url)

            if allowed_image(image.filename):
                filename = secure_filename(image.filename)

                image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

                print("Image saved")

                return redirect(request.url)

            else:
                print("That file extension is not allowed")
                return redirect(request.url)

    return render_template("public/upload_image.html")


# AI Chatbox incorperatred with the website
{"intents": [
        {"tag": "greeting",
         "patterns": ["Hi", "How are you", "Is anyone there?", "Hello", "Good day", "Whats up"],
         "responses": ["Hello!", "I am fine!", "Hi there, my name is bot 1. I am you PA today"],
         "context_set": ""
        },
        {"tag": "Help",
         "patterns": ["I have a problem", "How to upload videos", "Can you bring me to your boss "],
         "responses": ["What is your problem?", "If press the button, and should be able to press the button. ", "Sure, here is his email(rohitvijayakumar09@gmail.com)"
         "context_set": ""
        },
        {"tag": "personal"
         "patterns": ["How are you", "What is your favorite thing to do", "How old are you"],
         "responses": ["I am fine", "I like to code!", "I am currently the correct age to use discord"],
         "context_set": ""
        }
   ]
}


words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
        
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)
