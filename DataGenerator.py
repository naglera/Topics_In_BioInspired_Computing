import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def DataGenerator():
    # Read Data
    data = pd.read_csv('winequality-red.csv')
    # Create Matrix of Independent Variables
    X = data.drop(['quality'], axis=1)
    # Create Vector of Dependent Variable
    y = data['quality']
    # Create a Train Test Split for Genetic Optimization
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # Normalizing Data
    scaler = StandardScaler()
    X_trainu = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Put all labels between 0-5
    min_quality = min(np.min(y_train), np.min(y_test))
    y_train = y_train - min_quality
    y_test = y_test - min_quality
    return X_train, y_train, X_test, y_test
