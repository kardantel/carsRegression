"""
Created by kardantel at 8/20/2020
__author__ = 'Carlos Pimentel'
__email__ = 'carlosdpimenteld@gmail.com'
"""

from model import Models
from utils import Utils

model = Models()
utils = Utils()


def main(data):
    '''
    Main function.
    '''
    # Quantity, type and completeness of the available features are evaluated.
    new_data = utils.first_analysis(data)
    # Categorical variables are converted to numeric variables.
    new_data = utils.dummies(new_data)
    # The numerical variables and the target variable are separated.
    X, y = utils.features_dataset(new_data)
    # The data is prepared, scaled, the model is created, metrics are obtained,
    # and the prediction is made.
    model.data_preparation(X, y)
    model.get_metrics()


if __name__ == "__main__":

    cars = utils.load_from_csv('./in/Craiglist_Cars.csv')
    main(cars)
