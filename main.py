# -*- coding: utf-8 -*-
"""LSTM - 3char_to_1char.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11iRgS-N9rwc6UJM1tSxFKoRZQ96skrt-?usp=sharing
"""

# Naive LSTM to learn three-char time steps to one-char mapping  
import numpy  
from keras.models import Sequential  
from keras.layers import Dense  
from keras.layers import LSTM  
from keras.utils import np_utils

# fix random seed for reproducibility  
numpy.random.seed(7)

# define the raw dataset  
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  

# create mapping of characters to integers (0-25) and the reverse
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

''' enumerate() -> output index and element at the same time
{'a','b','c'} -> {0:'a', 1:'b', 2:'c'}  
'''

# prepare the dataset of input to output pairs encoded as integers 
seq_length = 3  
dataX = []  
dataY = []  

for i in range(0, len(alphabet) - seq_length, 1):  
    seq_in = alphabet[i:i + seq_length]  
    seq_out = alphabet[i + seq_length]  
    dataX.append([char_to_int[char] for char in seq_in])  
    dataY.append(char_to_int[seq_out])  
    print(seq_in, '->', seq_out)

# reshape X to be [samples, time steps, features]  
X = numpy.reshape(dataX, (len(dataX), seq_length, 1))  

# normalize  
XX = X / float(len(alphabet))  

# one hot encode the output variable  
y = np_utils.to_categorical(dataY)  

''' np_utils.to_categorical() -> 將類別向量轉換為二進位的矩陣類型表示
[0,1,2] -> [[1. 0. 0.] [0. 1. 0.] [0. 0. 1.]]
'''

print('dataX reshape:\n', X)

print('np_utils.to_categorical():\n', y)

print('X.shape:\n', X.shape)

# create and fit the model  
model = Sequential()  
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))  
model.add(Dense(y.shape[1], activation='softmax'))  
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
model.fit(X, y, epochs=500, batch_size=1, verbose=2)

# summarize performance of the model  
scores = model.evaluate(X, y, verbose=0)  
print("-----Model Accuracy: %.2f%%-----" % (scores[1]*100))

# demonstrate some model predictions  
for pattern in dataX:  
    x = numpy.reshape(pattern, (1, len(pattern), 1))  
    xx = x / float(len(alphabet))  
    prediction = model.predict(x, verbose=0)  
    index = numpy.argmax(prediction)  
    result = int_to_char[index]  
    seq_in = [int_to_char[value] for value in pattern]  
    print(seq_in, "->", result)

print('x.shape:\n', x.shape)

# demonstrate predicting random patterns  
print("-----Test a Random Pattern:-----")  
for i in range(0,20):  
    pattern_index = numpy.random.randint(len(dataX))  
    pattern = dataX[pattern_index]  
    x = numpy.reshape(pattern, (1, len(pattern), 1))  
    xx = x / float(len(alphabet))  
    prediction = model.predict(x, verbose=0)  
    index = numpy.argmax(prediction)  
    result = int_to_char[index]  
    seq_in = [int_to_char[value] for value in pattern]  
    print(seq_in, "->", result)

# Save the entire model to a h5 file.
model.save('./3char_to_1char.h5')

# demonstrate predicting input patterns
pattern_input = str(input('輸入英文字母(A-W)：')).upper()
pattern_index = char_to_int[pattern_input]
pattern = dataX[pattern_index]
x = numpy.reshape(pattern, (1, len(pattern), 1))  
xx = x / float(len(alphabet))  
prediction = model.predict(x, verbose=0)  
index = numpy.argmax(prediction)  
result = int_to_char[index]  
seq_in = [int_to_char[value] for value in pattern]  
print(seq_in, "->", result)