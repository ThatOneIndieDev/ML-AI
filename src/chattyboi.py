import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from pyexpat.errors import messages

from tensorflow.keras.models import load_model
from training import lemmatizer, intents

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# The following function below converts the sentence into a list of 1s amd 0s to input inside the model created

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# The following function below predicts what the class of the input will be after using the model to predict.
# Then we set an error treshold that if the prediction goes above this we can not not use those values.
# After that we sort the probabilities in reverse order and append the results in the return_list and return said return_list.

def predict_class(sentence):
    bag = bag_of_words(sentence)
    res = model.predict(np.array([bag]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1],reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_resopnse(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("Bot is running!")

while True:
    message= input("")
    ints = predict_class(message)
    res = get_resopnse(ints,intents)
    print(res)