import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Embedding, Flatten, Conv1D, MaxPooling1D, Activation
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
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
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
OPEN_WEATHER_KEY = "558b6d70c65894a8a901b86f730beead"
ERROR_THRESHOLD = 0.25


class ChatBot:
    def __init__(self, sound_on=False, face_recognition_on=False):
        self.model = self.get_model()
        self.lemmatizer = WordNetLemmatizer()
        self.intents = self.get_intents()
        self.words = self.get_words()
        self.classes = self.get_classes()
        self.sound_file_index = 0
        self.sound_file_name = f"spoken/voice{self.sound_file_index}.mp3"
        self.speech_recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.sentence_filter = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\'\t\n'
        self.max_sentence_len = 100
        self.batch_size = 32
        self.vocab_size = 0
        self.embedding_size = 128
        self.n_filters = 64
        self.n_epochs = 2
        self.lstm_output_size = 70
        self.pool_size = 4
        self.kernel_size = 5
        self.sound_on = sound_on
        self.face_recognition_on = face_recognition_on
        self.talking_to = None
        self.face_names = []
        self.face_locations = []
        self.process_frame = True

    @staticmethod
    def connectDB():
        return connect(host='localhost',
                       user='root',
                       password='',
                       db='bot',
                       charset='utf8mb4',
                       cursorclass=cursors.DictCursor)

    @staticmethod
    def geo_locate():
        g = geocoder.ip('me')
        lat, lon = g.latlng
        return lat, lon

    @staticmethod
    def get_weather(lat, lon):
        weather = requests.get(
            f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPEN_WEATHER_KEY}").json()
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

        # try using different optimizers and different optimizer configs
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

        print('Train...')
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

        print('Train...')
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

                    # tokenize each word
                    w = nltk.word_tokenize(pattern)
                    words.extend(w)
                    # add documents in the corpus
                    documents.append((w, intent['tag']))

                    # add to our classes list
                    if intent['tag'] not in classes:
                        classes.append(intent['tag'])

            # lemmaztize and lower each word and remove duplicates
            words = [self.lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
            words = sorted(list(set(words)))

            # sort classes
            classes = sorted(list(set(classes)))

            # documents = combination between patterns and intents
            print(len(documents), "documents")

            # classes = intents
            print(len(classes), "classes", classes)

            # words = all words, vocabulary
            print(len(words), "unique lemmatized words", words)

            with open(WORDS, 'wb') as f:
                pickle.dump(words, f)

            with open(CLASSES, 'wb') as f:
                pickle.dump(classes, f)

            # create our training data
            training = []

            # create an empty array for our output
            output_empty = [0] * len(classes)

            # training set, bag of words for each sentence
            for doc in documents:
                # initialize our bag of words
                bag = []

                # list of tokenized words for the pattern
                pattern_words = doc[0]

                # lemmatize each word - create base word, in attempt to represent related words
                pattern_words = [self.lemmatizer.lemmatize(word.lower()) for word in pattern_words]

                # create our bag of words array with 1, if word match found in current pattern
                for w in words:
                    bag.append(1) if w in pattern_words else bag.append(0)

                # output is a '0' for each tag and '1' for current tag (for each pattern)
                output_row = list(output_empty)
                output_row[classes.index(doc[1])] = 1

                training.append([bag, output_row])

            # shuffle our features and turn into np.array
            random.shuffle(training)
            training = np.array(training)

            # create train and test lists. X - patterns, Y - intents
            train_x = list(training[:, 0])
            train_y = list(training[:, 1])
            print("Training data created")

            # Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output
            # layer contains number of neurons
            # equal to number of intents to predict output intent with softmax
            model = Sequential()
            model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(len(train_y[0]), activation='softmax'))

            # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for
            # this model
            sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

            # fitting and saving the model
            hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
            model.save(model, hist)

            print("model created")

        return load_model(MODEL)

    def clean_up_sentence(self, sentence):
        # tokenize the pattern - split words into array
        sentence_words = nltk.word_tokenize(sentence)

        # stem each word - create short form for word
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

    def bow(self, sentence, show_details=True):
        # tokenize the pattern
        sentence_words = self.clean_up_sentence(sentence)

        # bag of words - matrix of N words, vocabulary matrix
        bag = [0] * len(self.words)
        for s in sentence_words:
            for i, w in enumerate(self.words):
                if w == s:
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)
        return np.array(bag)

    def predict_class(self, sentence):
        # filter out predictions below a threshold
        p = self.bow(sentence, show_details=False)
        res = self.model.predict(np.array([p]))[0]

        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        # sort by strength of probability
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

        # Display the resulting image
        # Initialize some variables

        while True:
            known_face_names = list(faces.keys())

            # Grab a single frame of video
            ret, frame = video_capture.read()

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
            if self.process_frame:
                # Find all the faces and face encodings in the current frame of video
                self.face_locations = fr.face_locations(rgb_small_frame)
                face_encodings = fr.face_encodings(rgb_small_frame, self.face_locations)

                for face_encoding in face_encodings:
                    for name in known_face_names:
                        faces_encoded = [faces[name]]
                        for f in faces_encoded:
                            # See if the face is a match for the known face(s)
                            matches = fr.compare_faces(f, face_encoding)

                            # # If a match was found in known_face_encodings, just use the first one.
                            # if True in matches:
                            #     first_match_index = matches.index(True)
                            #     name = known_face_names[first_match_index]

                            # Or instead, use the known face with the smallest distance to the new face
                            face_distances = fr.face_distance(f, face_encoding)
                            try:
                                best_match_index = np.argmin(face_distances)
                                if matches[best_match_index]:
                                    self.face_names.append(name)
                                    self.talking_to = name
                            except ValueError:
                                pass

            self.process_frame = not self.process_frame

            # Display the results
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, self.talking_to, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Video', frame)
            print(self.talking_to)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release handle to the webcam
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


def search_wikipedia(query):
    response = requests.get(f"https://en.wikipedia.org/wiki/{query}")
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.find_all('p')
    for paragraph in text:
        print(paragraph.text)


'''
def get_retinas():
    training = []
    for root, dirs, files in os.walk('retina training'):
        tmp = root.split('\\')
        if len(tmp) == 2:
            idn = tmp[1]
            retinas = [idn]
            for path in files:
                path = os.path.join(root, path)
                if path.split('.')[-1] == 'JPG':
                    retina = mpimg.imread(path)
                    small = cv2.resize(retina, (0, 0), fx=0.25, fy=0.25)
                    retinas.append(small)
            training.append(retinas)
        else:
            pass

    validation = []
    for root, dirs, files in os.walk('retina validation'):
        tmp = root.split('\\')
        if len(tmp) == 2:
            idn = tmp[1]
            retinas = [idn]
            for path in files:
                path = os.path.join(root, path)
                if path.split('.')[-1] == 'JPG':
                    retina = mpimg.imread(path)
                    small = cv2.resize(retina, (0, 0), fx=0.25, fy=0.25)
                    retinas.append(small)
            validation.append(retinas)
        else:
            pass

    random.shuffle(training)
    random.shuffle(validation)

    return np.array(training), np.array(validation)


training, validation = get_retinas()
X_train = []
y_train = []
X_test = []
y_test = []
for tRow, vRow in zip(training, validation):
    for item in tRow[1:]:
        X_train.append(item)
        y_train.append(tRow[0])
    for item in vRow[1:]:
        X_test.append(item)
        y_test.append(vRow[0])

print(np.array(X_train[0]).shape)

num_classes = 79

model = Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(340, 512, 3)),
  tf.keras.layers.Conv3D(16, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling3D(),
  tf.keras.layers.Conv3D(32, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling3D(),
  tf.keras.layers.Conv3D(64, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling3D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs = 10
history = model.fit(
  X_train,
  validation_data=X_test,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


T_SHIRT = 0
TROUSER = 1
PULLOVER = 2
DRESS = 3
COAT = 4
SANDAL = 5
SHIRT = 6
SNEAKER = 7
BAG = 8
ANKLE_BOOT = 9

class_names = ["t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]

data = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

train_images = train_images/255.0
test_images = test_images/255.0
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Tested accuracy: {test_acc}")

prediction = model.predict(test_images)
print(class_names[np.argmax(prediction[0])])
'''


if __name__ == "__main__":
    #ChatBot().talk()
    search_wikipedia("Kazan")
