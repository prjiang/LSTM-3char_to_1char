# LSTM-3char_to_1char

`Colab Code:` [data file](https://colab.research.google.com/drive/11iRgS-N9rwc6UJM1tSxFKoRZQ96skrt-?usp=sharing)

## A string summary of the network.

Define an LSTM network with 32 units and an output layer with a softmax activation function for making predictions.

Because this is a multi-class classification problem, log loss function(categorical_crossentropy) is used, and optimize the network using the ADAM optimization function.

The model is fit over 500 epochs with a batch size of 1.

`Reference:` [Understanding Stateful LSTM Recurrent Neural Networks in Python with Keras](https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/)

<br>

<hr>

<br>

## Tutorial

### 前置作業

`import` 需要使用到的函式。

``` python
import numpy  
from keras.models import Sequential  
from keras.layers import Dense  
from keras.layers import LSTM  
from keras.utils import np_utils 
```

<br>

`numpy.random.seed()` 一個記錄隨機亂數之容器，確保每次使用的隨機亂數皆相同。

```python
# fix random seed for reproducibility  
numpy.random.seed(7)
```

<br>

創建字母資料。

建立資料型態轉換之函式。
* 將字串轉為數值
* 將數值轉為字串

```python
# define the raw dataset  
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  

# create mapping of characters to integers (0-25) and the reverse
# Output index and element at the same time {'a','b','c'} -> {0:'a', 1:'b', 2:'c'}
char_to_int = dict((c, i) for i, c in enumerate(alphabet))  
int_to_char = dict((i, c) for i, c in enumerate(alphabet))  
```

<br>

查看字串轉換對應之數值。

```python
print(char_to_int)

''' output
{'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25}
'''
```

<br>

資料分割成以一組三個字母進行預測下一個字母。 `stride = 1`

將每組資料與其對應到的下一個字母轉換為數值，並加入至輸入與輸出之列表中。

```python
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
```

<br>

`np_utils.to_categorical()` 將類別向量轉換為二進位的矩陣類型表示。

> e.g. [0, 1, 2] -> [[1. 0. 0.] [0. 1. 0.] [0. 0. 1.]]

```python
# reshape X to be [samples, time steps, features]  
X = numpy.reshape(dataX, (len(dataX), seq_length, 1))  

# normalize  
XX = X / float(len(alphabet))  

# one hot encode the output variable  
y = np_utils.to_categorical(dataY)  
```

查看重新調整維度的訓練資料。

```python
print('dataX reshape:\n', X)
```

查看每一組輸出結果對應到之位置。

```python
print('np_utils.to_categorical():\n', y)
```

<br>

### 模型訓練

建立與訓練模型。

```python
# create and fit the model  
model = Sequential()  
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))  
model.add(Dense(y.shape[1], activation='softmax'))  
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
model.fit(X, y, epochs=500, batch_size=1, verbose=2)  
```

<br>

查看模型準確率。

顯示所有輸入資料之預測結果。

```python
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
```