import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Activation, Dense, LSTM
SEQ_LENGTH = 50
STEP_SIZE = 3

file_path = tf.keras.utils.get_file('shakespeare.txt', 
                                    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
file = open(file_path, 'rb').read().decode(encoding="utf-8").lower()

file = file[300000:800000]

characters = sorted(set(file))
char_ind = dict((c, i) for i, c in enumerate(characters))
ind_char = dict((i, c) for i, c in enumerate(characters))

sentences = []
next_char = []

for i in range(0, len(file) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(file[i: i + SEQ_LENGTH])
    next_char.append(file[i + SEQ_LENGTH])

x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=np.bool)
y = np.zeros((len(sentences), len(characters)), dtype=np.bool)

# training 
for i, satz in enumerate(sentences):
    for t, char in enumerate(satz):
        x[i, t, char_ind[char]] = 1
    y[i, char_ind[next_char[i]]] = 1

model = Sequential()
model.add(LSTM(128, 
               input_shape = (SEQ_LENGTH, len(characters)), 
               ))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))
model.fit(x, y, batch_size=256, epochs=4)

# saving model
# model.save('text-gen.keras') # to load for future use : tf.keras.models.load_model("model_name")

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate(length, temperature):
    start_ind = random.randint(0, len(file) - SEQ_LENGTH - 1)
    generated = ""
    sentence = file[start_ind: start_ind + SEQ_LENGTH]
    generated += sentence
    for i in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(characters)))
        for y, character in enumerate(sentence):
            x[0, t, char_ind[character]] = 1
        predict = model.predict(x, verbose=0)[0]
        next_index = sample(predict, temperature)
        next_character = ind_char[next_index]

        generated += next_character
        sentence = sentence[1:] + next_character
    return generated

print(generate(300, 0.2))