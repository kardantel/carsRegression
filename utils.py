'''
File that contains all the methods used by the classifiers.
'''

import warnings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
warnings.filterwarnings('ignore')


class Utils:
    '''
    Class containing all the methods used as support.
    '''
    @classmethod
    def load_from_csv(cls, path):
        '''
        Returns a list with the datasets from a specified address.

        ## Parameters
        path: address where the datasets are stored.
        '''
        return pd.read_csv(path)

    @classmethod
    def first_analysis(cls, data):
        '''
        Print various information about the original dataset.

        ## Parameters
        data: data obtained form load_from_csv method.
        '''
        # The null percentage is printed for each feature.
        print("Percentage of nulls for each feature:")
        num_null = 100 * data.isnull().sum() / data.shape[0]
        print(num_null)
        print()
        # The number of data types is printed.
        print("Number of data types:")
        types = pd.DataFrame(data.dtypes)
        print(types.groupby(0).size())
        print()
        # Counting of different values in each feature.
        print("Number of values per feature:")
        features = types.index[types[0] == 'O'].values
        for line in features:
            print(
                f'The feature {line} contains {str(len(data[line].unique()))} different values')
            # Null values in each column are filled with the mode of all values
            # in this column.
            if num_null.loc[line] < 10:
                data[line] = data[line].fillna(data[line].mode()[0])
        print()
        print("Check that number of nulls decreases:")
        print(100 * data.isnull().sum() / data.shape[0])

        return data

    @classmethod
    def dummies(cls, data, save=False, show=False):
        '''
        It transforms categorical variables to numeric with 'get_dummies'.

        ## Parameters
        data: data obtained form load_from_csv method.
        save: boolean [optional]
            Saves new dataset after use 'get_dummies' method.
        show: boolean [optional]
            Prints new dataset after use 'get_dummies' method.
        '''
        types = pd.DataFrame(data.dtypes)
        features = types.index[types[0] == 'O'].values
        df = data.copy()
        for col in features:
            df = pd.concat([df, (pd.get_dummies(df[col])).astype(int)], axis=1)
            df.drop(columns=[col], inplace=True)
        # When creating the dummies variables, several columns are created
        # referring to 'other' categories that do not add any value to the
        # dataset, therefore they are eliminated.
        df.drop('other', axis=1, inplace=True)

        if save:
            df.to_csv(f'./in/new_craiglist.csv', index=False)

        if show:
            print()
            print("New dataset after use 'get_dummies' method:")
            print(df.head())

        return df

    @classmethod
    def features_dataset(cls, data, show=False):
        '''
        The numerical variables and the target variable are separated.

        ## Parameters
        data: data obtained form load_from_csv method.
        show: boolean [optional]
            Prints new dataset after use 'get_dummies' method.
        '''
        types = pd.DataFrame(data.dtypes)
        numeric_columns = list(
            set(types.index[types[0] == "int64"].values) - set(["price"]))
        X = data[numeric_columns]
        y = data["price"]

        if show:
            print()
            print("Features dataset:")
            print(X.head())
            print()
            print("Target:")
            print(y.head())

        return X, y
