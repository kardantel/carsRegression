'''
File containing the code for AdaBoost.
'''

import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import pandas as pd
import random
warnings.filterwarnings('ignore')


class Models:
    '''
    Class that contains all the methods used for regression.
    '''

    @classmethod
    def prediction(cls):
        '''
        Make predictions with the model using the validation values.
        '''
        real = pd.DataFrame(cls.y_val)
        predic = cls.model.predict(cls.X_val_scaled)
        rescaled_values = cls.scaler2.inverse_transform(predic)
        pred_escal = pd.DataFrame(rescaled_values)
        # Shows actual values and predictions
        print()
        print("Making the prediction...")
        for i in range(0, 5):
            print("Real = %s, Prediction = %s" %
                  (real[0][i], pred_escal[0][i]))

    @classmethod
    def get_metrics(cls, show=False):
        '''
        Find the mean absolute error and loss values.

        ## Parameters
        show: boolean [optional]
            Print the metrics obtained with de neural network.
        '''
        result = cls.model.evaluate(cls.X_test_scaled, cls.y_test_scaled)
        for i in range(len(cls.model.metrics_names)):
            print()
            print("Metric", cls.model.metrics_names[i], ":", str(
                round(result[i], 2)))

        cls.prediction()

        if show:
            plt.figure(figsize=(13, 6))
            plt.plot(cls.history.history['loss'])
            plt.plot(cls.history.history['val_loss'])
            plt.title("Model losses with training set and tests by epoch")
            plt.ylabel('MSE')
            plt.xlabel('Epocs')
            plt.legend(['Training', 'Validation'], loc='upper right')
            plt.show()

    @classmethod
    def ann(cls, save=False):
        '''
        Scale the data to be used appropriately in the neural network.

        ## Parameters
        save: boolean [optional]
            Saves the model on 'model' folder.
        '''
        cls.model = Sequential()
        cls.model.add(
            Dense(256, input_dim=cls.X_train.shape[1], activation="relu"))
        cls.model.add(Dense(128, activation="relu"))
        cls.model.add(Dense(128, activation="relu"))
        # A DropOut regularization layer is included that turns off 20% of the
        # available neruones in each iteration.
        cls.model.add(Dropout(0.2))
        cls.model.add(Dense(1, activation="linear"))
        cls.model.compile(optimizer="adam", loss="mse",
                          metrics=["mean_absolute_error"])

        print()
        print("Training the model...")
        cls.history = cls.model.fit(cls.X_train_scaled, cls.y_train_scaled,
                                    validation_data=(
                                        cls.X_val_scaled, cls.y_val_scaled),
                                    epochs=50, batch_size=1024, verbose=0)
        if save:
            cls.model.save('./model/PricePrediction.h5')

        cls.get_metrics(show=True)

        return cls.history

    @classmethod
    def data_standard(cls):
        '''
        Scale the data to be used appropriately in the neural network.

        ## Parameters
        path: address where the datasets are stored.
        '''
        # The first climber is created based on the training characteristics
        # variable and with this the other information sets are scaled.
        cls.scaler1 = StandardScaler()
        cls.scaler1.fit(cls.X_train)
        cls.X_train_scaled = cls.scaler1.transform(cls.X_train)
        cls.X_val_scaled = cls.scaler1.transform(cls.X_val)
        cls.X_test_scaled = cls.scaler1.transform(cls.X_test)

        # A second scaler is created that will take the training objective
        # variable as input and with this the other information sets are scaled
        cls.scaler2 = StandardScaler()
        cls.scaler2.fit(cls.y_train)
        cls.y_train_scaled = cls.scaler2.transform(cls.y_train)
        cls.y_val_scaled = cls.scaler2.transform(cls.y_val)
        cls.y_test_scaled = cls.scaler2.transform(cls.y_test)

        cls.ann(save=True)

    @classmethod
    def data_preparation(cls, X, y):
        '''
        Separate into training, testing and validation data.

        ## Parameters
        path: address where the datasets are stored.
        '''
        # The variables in training and testing are separated.
        # 80% train, 20% test
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(X, y,
                                                                            test_size=0.2,
                                                                            random_state=2020)
        # The variables in effective training and validation are separated.
        # 90% train, 10% validation
        cls.X_train, cls.X_val, cls.y_train, cls.y_val = train_test_split(cls.X_train,
                                                                          cls.y_train,
                                                                          test_size=0.1,
                                                                          random_state=2020)

        # The size (shape) of the variables is changed. It goes from a row
        # vector to a column vector of (n, 1).
        cls.y_train = cls.y_train.values.reshape(-1, 1)
        cls.y_test = cls.y_test.values.reshape(-1, 1)
        cls.y_val = cls.y_val.values.reshape(-1, 1)

        cls.data_standard()
