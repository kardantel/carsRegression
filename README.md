# Vehicle price predictor with a neural network
This repository makes a prediction, with a linear regression, of the sale price of vehicles. The prediction is made with a Keras neural network. The code performs a first study of the dataset and standardizes the data to be used by the neural network. At the end, the loss metrics and the mean absolute error are shown, with the option to view the training and validation curves.

### Features

- Use Keras from Tensorflow in version 2.5.0.
- The loss metrics and the mean absolute error are shown.
- Uses the ** Craiglist_Cars.csv ** dataset.
- Shows different characteristics of the dataset such as number of variables or percentage of null values per variable.
- It allows to observe a graph with the training and validation curves.
- It shows 5 predictions made with 5 values taken from the validation set samples.

## How to use
Just execute in the terminal the code `python3 main.py` this will train the regression model.

### Result

An image with the training and validation curves that can be obtained is shown below:

![](https://i.imgur.com/dUqYwzi.png)
