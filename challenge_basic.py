"""
This Python file provides some useful code for reading the training file
"cleaned_data_combined.csv". You may adapt this code as you see fit. However,
keep in mind that the code provided does only basic feature transformations
to build a rudimentary kNN model in sklearn. Not all features are considered
in this code, and you should consider those features! Use this code
where appropriate, but don't stop here!
"""

import re
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

file_name = "cleaned_data_combined.csv"
random_state = 42

def to_numeric(s):
    """Converts string `s` to a float.

    Invalid strings and NaN values will be converted to float('nan').
    """

    if isinstance(s, str):
        s = s.replace(",", '')
        s = pd.to_numeric(s, errors="coerce")
    return float(s)

if __name__ == "__main__":

    df = pd.read_csv(file_name)

    # Select a subset of features for the baseline model
    selected_features = ["Q2: How many ingredients would you expect this food item to contain?",
                         "Q3: In what setting would you expect this food to be served? Please check all that apply",
                         "Q4: How much would you expect to pay for one serving of this food item?",
                         "Q6: What drink would you pair with this food item?"]


    # Prepare the data for training
    df = df[selected_features + ["Label"]]

    # Handle missing values
    df = df.fillna(0)

    # Encode categorical features (if necessary)
    for col in selected_features:
        if df[col].dtype == 'object':
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Convert categorical labels to numerical values
    df = pd.get_dummies(df, columns=["Label"], prefix="Label")

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=random_state)

    x = df.drop(columns=[col for col in df.columns if col.startswith("Label_")]).values
    y = df[[col for col in df.columns if col.startswith("Label_")]].values

    # Train-test split
    n_train = int(0.8 * len(df))
    x_train = x[:n_train]
    y_train = y[:n_train]

    x_test = x[n_train:]
    y_test = y[n_train:]

    # Train and evaluate a kNN classifier
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(x_train, y_train)
    train_acc = clf.score(x_train, y_train)
    test_acc = clf.score(x_test, y_test)
    print(f"{type(clf).__name__} train acc: {train_acc}")
    print(f"{type(clf).__name__} test acc: {test_acc}")
