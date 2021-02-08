from fairlearn.datasets import fetch_adult, fetch_bank_marketing
import numpy as np
import pandas as pd

def get_adult():
    adult = fetch_adult()
    X = adult["data"]
    y = adult["target"]

    X[:, [9,-1]] = X[:, [-1,9]] # puts sex into the last column

    # Erase nan values
    y = y[~np.isnan(X).any(axis=1)]
    X = X[~np.isnan(X).any(axis=1)]

    # transform to binary and set proper type
    X = X.astype('float')
    for i, lab in enumerate(y):
        if y[i] == "<=50K":
            y[i] = 0.
        else:
            y[i] = 1.
    y = y.astype('float')
    n, _ = X.shape
    return X, y

def get_german():
    df = pd.read_csv("german_dataset/german_numeric.csv")
    X = df.drop("Type", axis=1).to_numpy()[:,1:] # remove label and index columns
    y = df["Type"].to_numpy()


    X = X.astype("float")
    y = y.astype("float")

    n, _ = X.shape
    return X, y


def get_compas():
    pass
