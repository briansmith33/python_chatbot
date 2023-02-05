from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Embedding, Conv1D, MaxPooling1D, Activation
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing import sequence, text
from gtts import gTTS
from playsound import playsound
import speech_recognition as sr
import face_recognition as fr
import cv2
import nltk
from nltk.stem import WordNetLemmatizer
from pymysql import cursors, connect
import pickle
import numpy as np
import json
import random
import os.path
import requests
import geocoder
from conversions import Kelvin_to_Fahrenheit
from bs4 import BeautifulSoup
from dotenv import dotenv_values

config = dotenv_values(".env")

MODEL = "chatbot_model.h5"
INTENTS = "intents.json"
WORDS = "words.pkl"
CLASSES = "classes.pkl"
SARCASTIC_WORDS = "sarcasm_words.pkl"
SARCASM_MODEL = "sarcasm_model.h5"
SENTIMENT_MODEL = "sentiment_model.h5"
SENTIMENT_TRAINING = "sentiment_training.csv"
SENTIMENT_PICKLE = "sentiment.pkl"
SENTIMENT_WORDS = "senti_words.pkl"
FACES = "faces.pkl"
OPEN_WEATHER_KEY = config['OPEN_WEATHER_API_KEY']
ERROR_THRESHOLD = 0.25


class ChatBot:
    lemmatizer = WordNetLemmatizer()
    sound_file_index = 0
    sound_file_name = f"spoken/voice{sound_file_index}.mp3"
    speech_recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    sentence_filter = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\'\t\n'
    max_sentence_len = 100
    batch_size = 32
    vocab_size = 0
    embedding_size = 128
    n_filters = 64
    n_epochs = 2
    lstm_output_size = 70
    pool_size = 4
    kernel_size = 5
    talking_to = None
    face_names = []
    face_locations = []
    process_frame = True

    def __init__(self, sound_on=False, face_recognition_on=False):
        self.model = self.get_model()
        self.intents = self.get_intents()
        self.words = self.get_words()
        self.classes = self.get_classes()
        self.sound_on = sound_on
        self.face_recognition_on = face_recognition_on

    @staticmethod
    def connectDB():
        return connect(host=config['MYSQL_HOST'],
                       user=config['MYSQL_USER'],
                       password=config['MYSQL_PASS'],
                       db=config['MYSQL_DB'],
                       charset='utf8mb4',
                       cursorclass=cursors.DictCursor)

    @staticmethod
    def geo_locate():
        g = geocoder.ip('me')
        lat, lon = g.latlng
        return lat, lon

    @staticmethod
    def get_weather(lat, lon):
        open_weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPEN_WEATHER_KEY}"
        weather = requests.get(open_weather_url).json()
        temp = Kelvin_to_Fahrenheit(weather['main']['temp'])
        temp_min = Kelvin_to_Fahrenheit(weather['main']['temp_min'])
        temp_max = Kelvin_to_Fahrenheit(weather['main']['temp_max'])
        condition = weather['weather'][0]['description']
        pressure = weather['main']['pressure']
        humidity = weather['main']['humidity']
        wind_speed = weather['wind']['speed']
        sunrise = weather['sys']['sunrise']
        sunset = weather['sys']['sunset']
        wind_direction = weather['wind']['deg']
        return temp, temp_min, temp_max, condition, pressure, humidity, wind_speed, wind_direction, sunrise, sunset

    @staticmethod
    def search_wikipedia(query):
        response = requests.get(f"https://en.wikipedia.org/wiki/{query}")
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.find_all('p')
        for paragraph in text:
            print(paragraph.text)

    @staticmethod
    def get_intents():
        with open(INTENTS, 'r') as f:
            return json.load(f)

    @staticmethod
    def get_words():
        with open(WORDS, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def get_classes():
        with open(CLASSES, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def get_encoded_faces():
        encoded = {}
        for root, dirs, files in os.walk('faces'):
            tmp = root.split('\\')
            if len(tmp) == 2:
                name = tmp[1]
                faces = []
                for path in files:
                    path = os.path.join(root, path)
                    face = fr.load_image_file(path)
                    encoding = fr.face_encodings(face)
                    if encoding:
                        faces.append(encoding[0])
                    else:
                        pass
                encoded[name] = faces
            else:
                pass

        with open(FACES, 'wb') as f:
            pickle.dump(encoded, f)
        return encoded

    @staticmethod
    def get_sarc_train():
        for l in open('Sarcasm_Headlines_Dataset_v2.json', 'r'):
            yield json.loads(l)

    @staticmethod
    def get_sarc_test():
        for l in open('Sarcasm_Headlines_Dataset.json', 'r'):
            yield json.loads(l)

    def get_train_data(self):
        words = []
        headlines = []
        sarcasm = []
        data = self.get_sarc_train()
        for i in data:
            w = text.text_to_word_sequence(i['headline'], filters=self.sentence_filter, lower=True, split=' ')
            words.extend(w)
            sarcasm.append(i['is_sarcastic'])
            headlines.append(w)

        words = [self.lemmatizer.lemmatize(w) for w in words]
        words = sorted(list(set(words)))
        self.vocab_size = len(words)
        # pickle.dump(words, open('sarcasm_words.pkl', 'wb'))
        encoded = [list([words.index(word) for word in h if word in words]) for h in headlines]
        x_train = encoded
        y_train = sarcasm
        return x_train, y_train

    def get_test_data(self):
        words = []
        headlines = []
        sarcasm = []
        data = self.get_sarc_test()
        for i in data:
            w = text.text_to_word_sequence(i['headline'], filters=self.sentence_filter, lower=True, split=' ')
            words.extend(w)
            sarcasm.append(i['is_sarcastic'])
            headlines.append(w)

        words = [self.lemmatizer.lemmatize(w) for w in words]
        words = sorted(list(set(words)))
        encoded = [list([words.index(word) for word in h if word in words]) for h in headlines]
        x_test = encoded
        y_test = sarcasm

        return x_test, y_test

    def train_sarcasm_model(self):
        x_train, y_train = self.get_train_data()
        x_test, y_test = self.get_test_data()
        x_train = sequence.pad_sequences(x_train, maxlen=self.max_sentence_len)
        x_test = sequence.pad_sequences(x_test, maxlen=self.max_sentence_len)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        model = Sequential()
        model.add(Embedding(self.vocab_size, 128, input_length=self.max_sentence_len))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

        hist = model.fit(x_train, y_train,
                         batch_size=self.batch_size,
                         epochs=10,
                         validation_data=[x_test, y_test])
        model.save(SARCASM_MODEL, hist)

    def train_sentiment_model(self):
        with open(SENTIMENT_TRAINING, 'r') as f:
            data = f.read()
        scores = []
        tweets = []
        for item in data.splitlines():
            scores.append(item.split('","')[0].replace('"', ''))
            tweets.append(item.split('","')[5].replace('"', ''))
        scores = scores[:100000]
        tweets = tweets[:100000]
        zipped = zip(scores, tweets)
        zip_array = []
        for i in zipped:
            zip_array.append(list(i))
        random.shuffle(zip_array)
        unzipped_tweets = []
        scores = []
        for item in zip_array:
            scores.append(item[0])
            unzipped_tweets.append(item[1])
        new_scores = []
        for score in scores:
            if score == '0':
                score = '-1'
            elif score == '2':
                score = '0'
            else:
                score = '1'
            new_scores.append(score)
        words = []
        tokenized = []
        for tweet in unzipped_tweets:
            tweet = tweet.split(' ')
            new_tweet = ""
            for word in tweet:
                if word == '':
                    pass
                elif word[:4] == 'http':
                    pass
                elif 'www.' in word:
                    pass
                elif word[0] == '@':
                    pass
                else:
                    new_tweet += word + ' '
            w = text.text_to_word_sequence(new_tweet, filters=self.sentence_filter, lower=True,
                                           split=' ')
            words.extend(w)
            tokenized.append(w)

        words = [self.lemmatizer.lemmatize(w) for w in words]
        words = sorted(list(set(words)))
        with open(SENTIMENT_WORDS, 'wb') as f:
            pickle.dump(words, f)
        vocab = len(words)
        indices = []
        for t in tokenized:
            print(t)
            if not t:
                pass
            else:
                encoded = []
                for word in t:
                    if word in words:
                        encoded.append(words.index(word))
                indices.append(list(encoded))
        print(indices)
        x_train = indices[:int(len(indices) * 0.75)]
        y_train = new_scores[:int(len(indices) * 0.75)]
        x_test = indices[int(len(indices) * 0.75):]
        y_test = new_scores[int(len(indices) * 0.75):]

        try:
            data = zip((x_train, y_train), (x_test, y_test))
            with open(SENTIMENT_PICKLE, 'wb') as f:
                pickle.dump(data, f)
        except:
            pass

        x_train = sequence.pad_sequences(x_train, maxlen=self.max_sentence_len)
        x_test = sequence.pad_sequences(x_test, maxlen=self.max_sentence_len)

        model = Sequential()
        model.add(Embedding(vocab, self.embedding_size, input_length=self.max_sentence_len))
        model.add(Dropout(0.25))
        model.add(Conv1D(self.n_filters,
                         self.kernel_size,
                         padding='valid',
                         activation='relu',
                         strides=1))
        model.add(MaxPooling1D(pool_size=self.pool_size))
        model.add(LSTM(self.lstm_output_size))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        hist = model.fit([x_train], [y_train],
                         batch_size=self.batch_size,
                         epochs=self.n_epochs,
                         validation_data=(x_test, y_test))
        score, acc = model.evaluate(x_test, y_test, batch_size=self.batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)

        model.save(SENTIMENT_MODEL, hist)

    def get_model(self):
        if not os.path.isfile(MODEL):
            words = []
            classes = []
            documents = []
            ignore_words = ['?', '!']
            with open(INTENTS, 'r') as f:
                intents = json.load(f)

            for intent in intents['intents']:
                for pattern in intent['patterns']:

                    w = nltk.word_tokenize(pattern)
                    words.extend(w)
                    documents.append((w, intent['tag']))

                    if intent['tag'] not in classes:
                        classes.append(intent['tag'])

            words = [self.lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
            words = sorted(list(set(words)))

            classes = sorted(list(set(classes)))

            with open(WORDS, 'wb') as f:
                pickle.dump(words, f)

            with open(CLASSES, 'wb') as f:
                pickle.dump(classes, f)

            training = []

            output_empty = [0] * len(classes)

            for doc in documents:
                bag = []

                pattern_words = doc[0]

                pattern_words = [self.lemmatizer.lemmatize(word.lower()) for word in pattern_words]

                for w in words:
                    bag.append(1) if w in pattern_words else bag.append(0)

                output_row = list(output_empty)
                output_row[classes.index(doc[1])] = 1

                training.append([bag, output_row])

            random.shuffle(training)
            training = np.array(training)

            train_x = list(training[:, 0])
            train_y = list(training[:, 1])

            model = Sequential()
            model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(len(train_y[0]), activation='softmax'))

            sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

            hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
            model.save(model, hist)

        return load_model(MODEL)

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)

        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bow(self, sentence, show_details=True):
        sentence_words = self.clean_up_sentence(sentence)

        bag = [0] * len(self.words)
        for s in sentence_words:
            for i, w in enumerate(self.words):
                if w == s:
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)
        return np.array(bag)

    def predict_class(self, sentence):
        p = self.bow(sentence, show_details=False)
        res = self.model.predict(np.array([p]))[0]

        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return return_list

    def get_response(self, ints):
        tag = ints[0]['intent']
        list_of_intents = self.intents['intents']
        try:
            for i in list_of_intents:
                if i['tag'] == tag:
                    result = random.choice(i['responses'])
                    return result
        except:
            return "I don't understand"

    def sarcasm(self, sentence):
        with open(SARCASTIC_WORDS, 'rb') as f:
            words = pickle.load(f)

        sentence_words = text.text_to_word_sequence(sentence, filters=self.sentence_filter, lower=True, split=' ')
        sentence_words = [self.lemmatizer.lemmatize(word) for word in sentence_words]
        tokenized = [words.index(word) for word in sentence_words if word in words]

        model = load_model(SARCASM_MODEL)
        sentence = sequence.pad_sequences(list([tokenized]), maxlen=self.max_sentence_len)
        result = model.predict(sentence)

        return result

    def speak(self, sentence):
        tts = gTTS(text=sentence, lang='en')
        tts.save(self.sound_file_name)
        playsound(self.sound_file_name)
        self.sound_file_index += 1

    def get_audio(self):
        with self.microphone as source:
            audio = self.speech_recognizer.listen(source)
            said = ''
            try:
                said = self.speech_recognizer.recognize_google(audio)
            except:
                pass
        return said

    def respond(self, msg):
        ints = self.predict_class(msg)
        res = self.get_response(ints)
        return res

    def match_face(self):
        with open(FACES, 'rb') as f:
            faces = pickle.load(f)
        video_capture = cv2.VideoCapture(0)

        while True:
            known_face_names = list(faces.keys())

            ret, frame = video_capture.read()

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            rgb_small_frame = small_frame[:, :, ::-1]

            if self.process_frame:
                self.face_locations = fr.face_locations(rgb_small_frame)
                face_encodings = fr.face_encodings(rgb_small_frame, self.face_locations)

                for face_encoding in face_encodings:
                    for name in known_face_names:
                        faces_encoded = [faces[name]]
                        for f in faces_encoded:
                            matches = fr.compare_faces(f, face_encoding)
                            face_distances = fr.face_distance(f, face_encoding)
                            try:
                                best_match_index = np.argmin(face_distances)
                                if matches[best_match_index]:
                                    self.face_names.append(name)
                                    self.talking_to = name
                            except ValueError:
                                pass

            self.process_frame = not self.process_frame

            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, self.talking_to, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            cv2.imshow('Video', frame)
            print(self.talking_to)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    def talk(self):
        if self.face_recognition_on:
            self.match_face()

        while True:
            if self.sound_on:
                req = self.get_audio()
                res = self.respond(req)
                self.speak(res)
            else:
                req = input("Chat: ")
                res = self.respond(req)
                print(res)


if __name__ == "__main__":
    ChatBot().talk()
