from flask import Flask, render_template, request, url_for, flash, redirect
import sqlite3
from werkzeug.exceptions import abort
import random
import json
import pickle
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
from keras.models import load_model


app = Flask(__name__)

# app.config['SECRET_KEY'] = 'ewexrg2376rg278rwiuxg' 

    
    
lemmatizer = WordNetLemmatizer()
intents = json.loads(open(r'intents.json').read())

words = pickle.load(open(r'words.pkl', 'rb'))
classes = pickle.load(open(r'classes.pkl', 'rb'))
model = load_model(r'models/chatbot_model_trained_till_completed2.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words (sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class (sentence):
    bow = bag_of_words (sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes [r[0]], 'probability': str(r[1])})
    return return_list
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice (i['responses'])
            break
    return result

 
@app.route("/message", methods=(["POST"]))
def create():
    data=request.form
    message=data['message']
    ints = predict_class (message)
    res = get_response (ints, intents)
    print(res)
    return(res,200)

# while True:
#     message = input("")
         
#     ints = predict_class (message)
#     res = get_response (ints, intents)
#     print (res)
    
    
if __name__ == "__main__":
    print("GO! Bot is running!")
    app.run(host="0.0.0.0", port=5000,debug=True)