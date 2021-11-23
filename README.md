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

`import` 需要使用到的函式。

``` python
import numpy  
from keras.models import Sequential  
from keras.layers import Dense  
from keras.layers import LSTM  
from keras.utils import np_utils 
```

<br>

`numpy.random.seed()` 為一個記錄隨機亂數之容器，確保每次使用的隨機亂數皆相同。

```python
# fix random seed for reproducibility  
numpy.random.seed(7)
```

<br>

創建字母資料

建立資料型態轉換之函式
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
