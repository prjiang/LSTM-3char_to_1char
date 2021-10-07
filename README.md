# LSTM-3char_to_1char

## Summary of the network.

Define an LSTM network with 32 units and an output layer with a softmax activation function for making predictions.

Because this is a multi-class classification problem, log loss function(categorical_crossentropy) is used, and optimize the network using the ADAM optimization function.

The model is fit over 500 epochs with a batch size of 1.

`Colab Code:` [data file](https://colab.research.google.com/drive/11iRgS-N9rwc6UJM1tSxFKoRZQ96skrt-?usp=sharing)

`Reference:` [Understanding Stateful LSTM Recurrent Neural Networks in Python with Keras](https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/)
