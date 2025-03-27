"""Clean up CSV, create training matrix and test Logistic Regression"""
import itertools
import re
from pathlib import Path

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Union

from numpy.ma.core import shape
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

TRAINFILE = "cleaned_data_combined_modified.csv"
hyperparams_ab = {'Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)': [1.1051709180756477, 1.1051709180756477], 'Q2: How many ingredients would you expect this food item to contain?': [1.3976083313721242e-13, 0.6215642319337298], 'Q3: In what setting would you expect this food to be served? Please check all that apply': [13.336718587418781, 0.6174191599884395], 'Q4: How much would you expect to pay for one serving of this food item?': [0.00026754091885852554, 1.1051709180756477], 'Q5: What movie do you think of when thinking of this food item?': [0.0013121764345119399, 7.244156166740926], 'Q6: What drink would you pair with this food item?': [0.006595397401169781, 1.1051709180756477], 'Q7: When you think about this food item, who does it remind you of?': [0.2232578131743815, 1.1051709180756477], 'Q8: How much hot sauce would you add to this food item?': [3.056348578264306, 1.1051709180756477]}



def extract_vocab(df: pd.DataFrame, text_column: str) -> List[str]:
    """
    Extract a sorted vocabulary list from the specified text column using regex.

    Parameters:
        df (pd.DataFrame): DataFrame containing the text data.
        text_column (str): Column name to extract words from.

    Returns:
        List[str]: Sorted list of unique words found in the text column.
    """
    vocab_set = set()
    # Drop missing values and extract words from each text entry
    for text in df[text_column].dropna():
        words = re.findall(r'\b\w+\b', str(text).lower())
        vocab_set.update(words)
    return sorted(vocab_set)

def extract_categories(df: pd.DataFrame, option_column: str) -> List[str]:
    option_set = set()
    for options in df[option_column].dropna():
        # Split the string by commas and add each option to the set after stripping whitespace
        for option in options.split(','):
            option_set.add(option.strip())
    return list(option_set)


def extract_number(value: Union[str, float, None]) -> float:
    """
    Extracts a number from messy text-based inputs.

    If the string represents a range (e.g., "5-10" or "12 to 15"), the average is returned.

    Parameters:
        value (str or any): Input value from which to extract the number.

    Returns:
        float: Extracted number or 0 if no valid number is found.
    """
    if pd.isna(value) or not isinstance(value, str):
        return 0

    # Extract all numeric sequences
    numbers = re.findall(r'\d+', value)
    if not numbers:
        return 0

    # Check for range indicators and compute the average if present
    if '-' in value or 'to' in value:
        num_list = [int(num) for num in numbers]
        return sum(num_list) / len(num_list)

    # Otherwise, return the first found number as float
    return float(numbers[0])


def create_text_features(df: pd.DataFrame, column: str, train_percent = 0.8) -> np.ndarray:
    """
    Converts a text column into a binary feature matrix based on the vocabulary.

    Parameters:
        df (pd.DataFrame): DataFrame containing the text data.
        column (str): The column name to process.

    Returns:
        np.ndarray: A binary matrix where each row represents an observation and each column
                    represents whether the corresponding vocabulary word appears in the text.
    """
    train_data = df.head(int(len(df)*train_percent))
    vocab = extract_vocab(train_data, column)
    word_to_index: Dict[str, int] = {word: idx for idx, word in enumerate(vocab)}
    N = len(df)
    V = len(vocab)
    X_bin = np.zeros((N, V), dtype=int)

    # Iterate through each text entry to populate the binary feature matrix
    for i, text in enumerate(df[column].fillna('')):
        words = re.findall(r'\b\w+\b', str(text).lower())
        for word in words:
            idx = word_to_index.get(word)
            if idx is not None:
                X_bin[i, idx] = 1
    return X_bin

def create_category_features(df: pd.DataFrame, column: str) -> np.ndarray:
    categories = extract_categories(df, column)
    cat_to_index: Dict[str, int] = {cat: idx for idx, cat in enumerate(categories)}
    N = len(df)
    V = len(categories)
    X_bin = np.zeros((N, V), dtype=int)
    # Iterate through each text entry to populate the binary feature matrix
    for i, options in enumerate(df[column].fillna('')):
        for option in options.split(','):
            option = option.strip()
            if option in cat_to_index:
                X_bin[i, cat_to_index[option]] = 1
    return X_bin

def create_bayes_features(df: pd.DataFrame, column: str, train_percent: float, hyperparams_ab: dict) -> np.ndarray:
    X = create_text_features(df, column, train_percent)
    t = create_t(df)
    n_train = int(train_percent * X.shape[0])
    X_train, t_train = X[:n_train], t[:n_train]
    a, b = hyperparams_ab[column][0], hyperparams_ab[column][1]
    pi, theta = naive_bayes_map(X_train, t_train, a_feat=a, b_feat=b)
    return compute_nb_probabilities(X, pi, theta)


def naive_bayes_map(X, t, n_labels=3, a_class=1.0, a_feat=1.0, b_feat=3.0):
    """
    Compute MAP estimates for a multiclass Bernoulli Naive Bayes model,
    allowing hyperparameters for the class prior and the Beta prior on features.

    Parameters:
        X: Binary feature matrix [N, V]
        t: Integer label vector [N]
        n_labels: Number of labels/classes
        a_class: Hyperparameter for class priors (Dirichlet prior).
        a_feat: Hyperparameter α for the Beta(α, β) feature prior
        b_feat: Hyperparameter β for the Beta(α, β) feature prior

    Returns:
        pi: Array of shape [n_labels], the MAP class priors
        theta: Array of shape [V, n_labels], the MAP feature probabilities
    """
    N, V = X.shape
    K = n_labels

    # One-hot encode labels
    Y = np.zeros((N, K))
    Y[np.arange(N), t] = 1

    # Count of examples in each class
    Nk = Y.sum(axis=0)  # shape [K]

    # MAP estimate of class priors with Dirichlet(alpha_class)
    pi = (Nk + a_class ) / (N + K * a_class)

    # MAP estimate of Bernoulli parameters with Beta(alpha_feat, beta_feat)
    # For each class k and feature v:
    #   θ_{v,k} = ( # of times feature v=1 in class k + alpha_feat ) / ( Nk + alpha_feat + beta_feat )
    theta = (X.T.dot(Y) + a_feat) / (Nk + a_feat + b_feat)

    return pi, theta

def compute_nb_probabilities(X, pi, theta):
    """
    Compute predicted class probabilities using the Naive Bayes model.

    Parameters:
        X: Binary feature matrix [N, V]
        pi: Class prior probabilities [K]
        theta: Feature likelihoods [V, K]

    Returns:
        probs: Array of predicted probabilities [N, K]
    """
    # Use log probabilities for numerical stability.
    log_probs = np.dot(X, np.log(theta)) + np.dot(1 - X, np.log(1 - theta)) + np.log(pi)
    # Convert log probabilities back to probabilities using softmax
    probs = np.exp(log_probs - np.max(log_probs, axis=1, keepdims=True))
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return probs


def create_X_t(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates feature matrix X and label matrix t from the CSV file.

    The function processes several columns:
      - Converts specific columns to numerical features using extract_number.
      - Converts text columns into binary indicator matrices.
      - Constructs a one-hot encoded label matrix.

    Parameters:
        filename (str): Path to the CSV file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Feature matrix X and label matrix t.
    """
    df = pd.read_csv(filename)

    # Define column names (assumes these exact names exist in the CSV)
    cols = {
        "id": "id",
        "complexity": "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
        "ingredients": "Q2: How many ingredients would you expect this food item to contain?",
        "setting": "Q3: In what setting would you expect this food to be served? Please check all that apply",
        "price": "Q4: How much would you expect to pay for one serving of this food item?",
        "movie": "Q5: What movie do you think of when thinking of this food item?",
        "drink": "Q6: What drink would you pair with this food item?",
        "reminder": "Q7: When you think about this food item, who does it remind you of?",
        "hotsauce": "Q8: How much hot sauce would you add to this food item?",
        "label": "Label"
    }

    N = len(df)

    # Process first two numerical columns (complexity and ingredients)
    complexity = df[cols["complexity"]].fillna(0).to_numpy().reshape(N, 1)
    ingredients = df[cols["ingredients"]].apply(extract_number).to_numpy().reshape(N, 1)
    X = np.hstack((complexity, ingredients))

    # Process 'setting' text column into binary features
    setting_features = create_text_features(df, cols["setting"])
    X = np.hstack((X, setting_features))

    # Process price column (numerical extraction)
    price = df[cols["price"]].apply(extract_number).to_numpy().reshape(N, 1)
    X = np.hstack((X, price))

    # Process 'movie' text column into binary features
    movie_features = create_text_features(df, cols["movie"])
    X = np.hstack((X, movie_features))

    # Process 'drink' text column into binary features
    drink_features = create_text_features(df, cols["drink"])
    X = np.hstack((X, drink_features))

    # Process 'reminder' text column into binary features
    reminder_features = create_text_features(df, cols["reminder"])
    X = np.hstack((X, reminder_features))

    # Process 'hotsauce' text column into binary features
    hotsauce_features = create_text_features(df, cols["hotsauce"])
    X = np.hstack((X, hotsauce_features))

    # Process labels into a one-hot encoded matrix
    labels = df[cols["label"]].fillna("").astype(str)
    unique_labels = sorted({label.strip() for label in labels})
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    t = np.zeros((N), dtype=int)
    for i, label in enumerate(labels):
        label_clean = label.strip()
        if label_clean in label_mapping:
            t[i] = label_mapping[label_clean]

    return X, t

def create_t(df: pd.DataFrame, label="Label"):
    labels = df[label].fillna("").astype(str)
    N = len(df)
    unique_labels = sorted({label.strip() for label in labels})
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    t = np.zeros((N), dtype=int)
    for i, label in enumerate(labels):
        label_clean = label.strip()
        if label_clean in label_mapping:
            t[i] = label_mapping[label_clean]
    return t

def create_X_t_selection(df, num_cols: list[str], bayes_cols: list[str], text_cols: list[str], category_cols: list[str],
                         train_percent:float, hyperparams_ab) -> Tuple[np.ndarray, np.ndarray]:
    N = len(df)
    X_parts = []  # Collect all features here

    # Process numeric columns
    for col in num_cols:
        num_features = df[col].apply(extract_number).to_numpy().reshape(N, 1)
        X_parts.append(num_features)

    # Process text columns
    for col in text_cols:
        text_features = create_text_features(df, col, train_percent)
        X_parts.append(text_features)

    # Process Bayesian columns
    for col in bayes_cols:
        bayes_features = create_bayes_features(df, col, train_percent, hyperparams_ab)
        X_parts.append(bayes_features)

    for col in category_cols:
        cate_features = create_category_features(df, col)
        X_parts.append(cate_features)

    # Concatenate all features horizontally
    if X_parts:
        X = np.hstack(X_parts)
    else:
        # Create an empty array with the appropriate number of rows and 0 columns.
        X = np.empty((df.shape[0], 0))

    # Create target vector
    t = create_t(df, "Label")

    return X, t


if __name__ == "__main__":
    train_percent = 0.8 # 0-1

    feature_columns = [
        "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
        "Q2: How many ingredients would you expect this food item to contain?",
        "Q3: In what setting would you expect this food to be served? Please check all that apply",
        "Q4: How much would you expect to pay for one serving of this food item?",
        "Q5: What movie do you think of when thinking of this food item?",
        "Q6: What drink would you pair with this food item?",
        "Q7: When you think about this food item, who does it remind you of?",
        "Q8: How much hot sauce would you add to this food item?"
    ]
    candidate_num_cols = [
        "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
        "Q2: How many ingredients would you expect this food item to contain?",
        "Q4: How much would you expect to pay for one serving of this food item?",
    ]
    candidate_text_cols = [
        # "Q3: In what setting would you expect this food to be served? Please check all that apply",
        # "Q5: What movie do you think of when thinking of this food item?",
        # "Q6: What drink would you pair with this food item?",
        # "Q7: When you think about this food item, who does it remind you of?",
        # "Q8: How much hot sauce would you add to this food item?"
    ]
    candidate_category_cols = [
        "Q3: In what setting would you expect this food to be served? Please check all that apply",
        "Q5: What movie do you think of when thinking of this food item?",
        "Q6: What drink would you pair with this food item?",
        "Q7: When you think about this food item, who does it remind you of?",
        "Q8: How much hot sauce would you add to this food item?"
    ]
    candidate_bayes_cols = [
        # "Q3: In what setting would you expect this food to be served? Please check all that apply",
        "Q5: What movie do you think of when thinking of this food item?",
        "Q6: What drink would you pair with this food item?",
        # "Q7: When you think about this food item, who does it remind you of?",
        # "Q8: How much hot sauce would you add to this food item?"
    ]



    file_name = "cleaned_data_combined.csv"
    data_path = Path.cwd().parent / file_name
    df = pd.read_csv(data_path)
    X, t = create_X_t_selection(df, candidate_num_cols, candidate_bayes_cols, candidate_text_cols,candidate_category_cols, train_percent, hyperparams_ab)

    # Use NumPy to split the data into 80% training and 20% testing sets
    N = X.shape[0]
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)

    train_size = int(train_percent * N)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    best_score = -np.inf
    best_combo = None
    c = 0
    for num_subset in itertools.chain.from_iterable(
            itertools.combinations(candidate_num_cols, r) for r in range(0, len(candidate_num_cols) + 1)
    ):
        for text_subset in itertools.chain.from_iterable(
                itertools.combinations(candidate_text_cols, r) for r in range(0, len(candidate_text_cols) + 1)
        ):
            for bayes_subset in itertools.chain.from_iterable(
                    itertools.combinations(candidate_bayes_cols, r) for r in range(0, len(candidate_bayes_cols) + 1)
            ):
                for category_subset in itertools.chain.from_iterable(
                        itertools.combinations(candidate_category_cols, r) for r in
                        range(0, len(candidate_category_cols) + 1)
                ):
                    if not (num_subset or text_subset or bayes_subset or category_subset):
                        continue
                    # Build feature matrix using your helper function
                    X, t = create_X_t_selection(
                        df,
                        list(num_subset),
                        list(bayes_subset),
                        list(text_subset),
                        list(category_subset),
                        train_percent,
                        hyperparams_ab
                    )

                    X_train, X_test = X[train_indices], X[test_indices]
                    t_train, t_test = t[train_indices], t[test_indices]

                    # Initialize and train the classifier
                    test_model = RandomForestClassifier(n_estimators=10, random_state=42)
                    test_model.fit(X_train, t_train)

                    # Evaluate the model on the test set
                    score = test_model.score(X_test, t_test)
                    if score > best_score:
                        best_score = score
                        best_combo = (num_subset, text_subset, bayes_subset, category_subset)
                        print("New best score:", best_score, "with combo:", best_combo)
                    if c % 10 == 0:
                        print("Iteration:", c)
                    c += 1

    X_train, X_test = X[train_indices], X[test_indices]
    t_train, t_test = t[train_indices], t[test_indices]
    print("Best CV accuracy:", best_score)
    print("Best feature combination:")
    for i in best_combo: print(i)

    print("Training feature matrix shape:", X_train.shape)
    print("Testing feature matrix shape:", X_test.shape)
    print("Training label matrix shape:", t_train.shape)
    print("Testing label matrix shape:", t_test.shape)

    X_train, X_test = X[train_indices], X[test_indices]
    t_train, t_test = t[train_indices], t[test_indices]

    # Initialize the logistic regression model
    test_model = LogisticRegression(max_iter=10000)

    # Train the model on the training data
    test_model.fit(X_train, t_train)

    # Predict on the test set
    y_pred = test_model.predict(X_test)

    # Evaluate the model's accuracy
    accuracy = accuracy_score(t_test, y_pred)
    print("LR Test Accuracy:", accuracy)
    print("LR Training Accuracy:", test_model.score(X_train, t_train))

    test_model = RandomForestClassifier(n_estimators=10, random_state=42)

    # Train the model on the training data
    test_model.fit(X_train, t_train)

    # Predict on the test set
    y_pred = test_model.predict(X_test)

    # Evaluate the model's accuracy
    accuracy = accuracy_score(t_test, y_pred)
    print("RF Test Accuracy:", accuracy)
    print("RF Training Accuracy:", test_model.score(X_train, t_train))


    # Train model on complete dataset
    final_model = LogisticRegression(max_iter=10000)
    final_model.fit(X, t)



    # Save weights in weights.txt
    np.savetxt("weights.txt", final_model.coef_, fmt="%.7f")  # Saves with 7 decimal places
    np.savetxt("bias.txt", final_model.intercept_, fmt="%.7f")
