import pandas as pd

def load(path):
    return pd.read_csv(path, header = 0)

def get_features_labels(df):
    return df.iloc[:,:-1], df.iloc[:,-1]

def clean():
    pass

def normalize():
    pass