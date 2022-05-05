from gtts import gTTS
import speech_recognition as sr
from playsound import playsound
import numpy as np
import random
from string import punctuation
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
from textblob import TextBlob


def import_tensorflow():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(0)
    import logging
    tf.get_logger().setLevel(logging.ERROR)
    return tf


tf = import_tensorflow()
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence, text

lemmatizer = WordNetLemmatizer()
model = load_model('assets/chatbot_model.h5')
intents = json.loads(open('assets/intents.json').read())
words = pickle.load(open('assets/words.pkl', 'rb'))
classes = pickle.load(open('assets/classes.pkl', 'rb'))


def sentiment(sentence):
    blob = TextBlob(sentence)
    sent = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    true_value = (1 - subjectivity) * sent
    return true_value


def sarcasm(sentence):
    with open('assets/sarcasm_words.pkl', 'rb') as f:
        Words = pickle.load(f)
    sentence_words = text.text_to_word_sequence(sentence, filters=f'{punctuation}\t\n', lower=True, split=' ')
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    tokenized = []
    for word in sentence_words:
        if word in Words:
            tokenized.append(Words.index(word))

    sarc_model = load_model('assets/sarcasm_model.h5')
    sentence = sequence.pad_sequences(list([tokenized]))
    result = sarc_model.predict(sentence)
    return result[0][0]


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)


def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


def speak(sentence):
    tts = gTTS(text=sentence, lang='en')
    filename = 'spoken/voice.mp3'
    tts.save(filename)
    playsound(filename)


def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        said = ''
        try:
            said = r.recognize_google(audio)
        except:
            pass
    return said


def query_dictionary(word):
    data_file = open('assets/dictionary_alpha_arrays.json').read()
    dictionary = json.loads(data_file)
    word = word.lower()
    letter = word[0]
    index = ord(letter) - 97
    try:
        print(dictionary[index][word])
    except:
        print("Sorry, that is not a word.")


if __name__ == "__main__":
    while True:
        user_response = input("Chatty> ")
        print(sarcasm(user_response))
        print(sentiment(user_response))
        chat_response = chatbot_response(user_response)
        print(chat_response)


