from threading import activeCount

import nltk
import random
import json
import numpy as np
import pickle
from nltk.stem import WordNetLemmatizer


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.utils.version_utils import training
from tensorflow.python.types.doc_typealias import document

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?','!', '.' , ',']

for intent in intents['intents']: #Now the intents file is our python dictionary that we will iterate over
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern) #Spliiting up indicvisual words
        words.append(word_list)
        documents.append((word_list, intent['tag'])) # tag marks the catagory of the word_list being appended
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

flat_words = [word for sublist in words for word in sublist] #Flattening the words list for lemmetization
words = [lemmatizer.lemmatize(word) for word in flat_words if word not in ignore_letters]
words = sorted(set(words)) # Quick way to remove duplicates in the code

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

#feeding the NN with the words converted into numerical values
# Going to set a word to either 1 or 0 if the specific word appears in the pattern and the will do the same with the classes.

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row]) #Created a trining list

# Pre-processing

random.shuffle(training)
# training = np.array(training,dtype=object)
#
# train_x = list(training[:,0])
# train_y = list(training[:,1])

train_x = [row[0] for row in training]
train_y = [row[1] for row in training]

model  = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu')) # 1st layer of our NN.
model.add(Dropout(0.5)) # For overfitting and not let dead neurons happen
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y),epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print("Done")
