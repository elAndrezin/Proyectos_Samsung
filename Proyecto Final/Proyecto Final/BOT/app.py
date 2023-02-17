#!/usr/bin/env python
# Este archivo usa el encoding: utf-8
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import os
from flask import send_from_directory
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import random
import json
import pickle
import numpy as np
import nltk
from pydub import AudioSegment
nltk.download('punkt')
import nltk
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from nltk.stem import SnowballStemmer
from flask import jsonify



app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# Carga el modelo en una variable global
model = load_model("model.h5")
# Load the tokenizer and classes
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))


@app.route('/')
def index():
    # Traductor

    # Página principal
    return render_template('index.html')


@app.route('/about')
def about():
    # Página Por que este traductor
    return render_template('About.html')

@app.route('/porque')
def porque():
    # Página Por que este traductor
    return render_template('Porque.html')


stemmer = SnowballStemmer('spanish')
ignore_words = ["?","¿","!","¡"]
data_file = open("intents.json", encoding='utf-8').read()
intents = json.loads(data_file)


# Crear un diccionario para convertir las palabras a su representación numérica
training = []
output_empty = [0] * len(classes)

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return np.array(bag)

ERROR_THRESHOLD = 0.8

@app.route('/predict', methods=['POST'])
def predict():

    # Obtener el mensaje del usuario
    message = request.form.get('message')
    # Imprimir el mensaje
    print(message)
    # El resto del código aquí ...
    # Convertir el patrón de diálogo a su representación numérica
    results = model.predict(np.array([bag_of_words(message, words)]))[0]
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    if len(results)>0:
         # Recibir la intención con mayor probabilidad
        class_id = results[0][0]  
        for intent in intents['intents']:
                    if intent['name'] == classes[class_id]:
                        # Recibir una respuesta aleatoria de la intención
                        print(random.choice(intent['responses'])) 
                        prediccion = random.choice(intent['responses'])            
                        response = {
                            'response': prediccion
                             }
    else:
        response = {
            'response': 'I am sorry, I dont understand you'

        }                   
    return jsonify(response)


@app.route("/save-recording", methods=["POST"])
def save_recording():
    audio = request.files["audio"]
    current_directory = os.path.dirname(os.path.abspath(__file__))
    audio.save(os.path.join(current_directory, "audio_grabado.wav"))
    return "Audio guardado exitosamente"

@app.route('/list', methods=['POST'])
def receive_order():
    order = request.form['order']
    # Remove text before and after the order
    order = order.split("es ")[1]
    order = order.split(".")[0]
    order = order.replace("un ","") # remove the word "un" 
    order = order.replace("una ","") # remove the word "una"
    order = order.strip()
    # Do something with the order, such as saving it to a file
    with open('orders.txt', 'a') as file:
        file.write(order + '\n')
    return "Order received"


if __name__ == '__main__':
    app.run(debug=False, threaded=False)
