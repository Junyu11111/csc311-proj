import re
import pandas as pd
import numpy as np
from pathlib import Path


def select_features(df, feature_cols, label_col):
    """
    Select and preprocess multiple text features for the model.
    For text features, this function converts the selected columns to strings
    and concatenates them into a single text field called 'combined_text'.

    Parameters:
        df: DataFrame.
        feature_cols: List of column names to use as features (text-based).
        label_col: Column name for the target label.

    Returns:
        A new DataFrame with two columns:
          - combined_text: the concatenation of the selected feature columns.
          - label: the original label.
    """
    # Convert all selected feature columns to strings
    for col in feature_cols:
        df[col] = df[col].astype(str)

    # Combine selected features into one text column.
    df["combined_text"] = df[feature_cols].apply(lambda row: " ".join(row), axis=1)

    # Return only the combined text and the label column.
    return df[["combined_text", label_col]]

def extract_vocab(df, text_column):
    """
    Extract a vocabulary list from the specified text column using regex.
    """
    vocab_set = set()
    for response in df[text_column].dropna():
        words = re.findall(r'\b\w+\b', response.lower())
        vocab_set.update(words)
    return list(sorted(vocab_set))

def make_bow(data, vocab, label_mapping=None):
    """
    Build a bag-of-words representation for a list of (text, label) pairs and map labels to integers.

    Parameters:
        data: List of (text, label) pairs.
        vocab: List of words to be used as features.
        label_mapping: Optional dictionary mapping label strings to integers.

    Returns:
        X: Binary feature matrix of shape [N, len(vocab)]
        t: Integer label vector of shape [N]
        label_mapping: Dictionary mapping label strings to integers.
    """
    N = len(data)
    V = len(vocab)
    X = np.zeros((N, V))

    if label_mapping is None:
        unique_labels = sorted({label.strip() for _, label in data})
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    t = np.zeros(N, dtype=int)
    word_to_index = {word: idx for idx, word in enumerate(vocab)}

    for i, (text, label) in enumerate(data):
        # text into words (split by whitespace)
        words_in_text = set(text.split())
        for word in words_in_text:
            word = word.strip().lower()
            if word in word_to_index:
                X[i, word_to_index[word]] = 1
        t[i] = label_mapping[label.strip()]

    return X, t, label_mapping

def naive_bayes_map(X, t, n_labels):
    """
    Compute MAP estimates for a multiclass Naive Bayes model.

    Parameters:
        X: Binary feature matrix [N, V]
        t: Integer label vector [N]
        n_labels: Number of labels

    Returns:
        pi: Array of shape [n_labels] representing class priors.
        theta: Array of shape [V, n_labels] representing feature likelihoods.
    """
    N, V = X.shape
    K = n_labels

    # One-hot encode labels
    Y = np.zeros((N, K))
    Y[np.arange(N), t] = 1

    Nk = Y.sum(axis=0)  # Count per class
    pi = (Nk + 1) / (N + K)
    theta = (X.T.dot(Y) + 1) / (Nk + 2)

    return pi, theta


def make_prediction(X, pi, theta):
    """
    Predict class labels for each example in X using the Naive Bayes model.

    Parameters:
        X: Binary feature matrix [N, V]
        pi: Class prior probabilities [K]
        theta: Feature likelihoods [V, K]

    Returns:
        predictions: Array of predicted class indices.
    """
    # Use log probabilities for numerical stability.
    log_probs = np.dot(X, np.log(theta)) + np.dot(1 - X, np.log(1 - theta)) + np.log(pi)
    predictions = np.argmax(log_probs, axis=1)
    return predictions

def accuracy(y_pred, y_true):
    """Compute prediction accuracy."""
    return np.mean(y_pred == y_true)

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

def integrate_nb_predictions(df, X, pi, theta, label_mapping, feature_name):
    """
    Integrate Naive Bayes predicted probabilities as new features into the original DataFrame.

    Parameters:
        df: Original DataFrame.
        X: Binary feature matrix [N, V].
        pi: Class prior probabilities [K].
        theta: Feature likelihoods [V, K].
        label_mapping: Dictionary mapping label strings to integers.

    Returns:
        df_with_nb: DataFrame with Naive Bayes probabilities appended as new features.
    """
    # Compute predicted probabilities
    nb_probs = compute_nb_probabilities(X, pi, theta)

    # Convert probabilities to a DataFrame
    nb_prob_df = pd.DataFrame(
        nb_probs,
        columns=[f"nb_prob_{label}_{feature_name}" for label in label_mapping.keys()]
    )

    # Append probabilities to the original DataFrame
    df_with_nb = pd.concat([df, nb_prob_df], axis=1)
    return df_with_nb

def text_to_one_hot_vocab(df, feature_columns, label_column):
    """
    Convert text features into one-hot encoded features and append them to the original DataFrame.

    Parameters:
        df: Pandas DataFrame
        feature_columns: List of keys for text feature columns.
        label_column: Label column.

    Returns:
        model_df: Modified Pandas DataFrame with added columns.
        vocabs: List of all the words appeared.

    Example:
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "feature1": ["a c", "a"],
    ...     "feature2": ["a", "b c"],
    ...     "label": ["pizza", "sushi"]
    ... })
    >>> feature_columns = ["feature1", "feature2"]
    >>> label_column = "label"
    >>> modified_df, vocab = text_to_one_hot_vocab(df, feature_columns, label_column)
    >>> print(modified_df)
      feature1 feature2  label    a    b    c
    0      a c        a  pizza  1.0  0.0  1.0
    1        a      b c  sushi  1.0  1.0  1.0
    """
    df_selected = select_features(df.copy(), feature_columns, label_column)
    print(df_selected)

    vocab = extract_vocab(df_selected, "combined_text")

    data_pairs = list(zip(df_selected["combined_text"].tolist(), df_selected[label_column].tolist()))

    X, t, label_mapping = make_bow(data_pairs, vocab)

    bow_df = pd.DataFrame(X, columns=vocab)

    modified_df = pd.concat([df, bow_df], axis=1)
    return modified_df, vocab

def text_to_numeric(df, feature_columns, label_column):
    """
    Convert text features into numeric Naive Bayes probability features and append them to the original DataFrame.

    Parameters:
        df: Pandas DataFrame.
        feature_columns: List of column names to use as text features.
        label_column: Column name for the target label.

    Returns:
        model_df: Modified Pandas DataFrame with added columns.
        vocabs: List of all the words appeared.

    Example:
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "feature1": ["a c", "a", "b c"],
    ...     "feature2": ["b", "a c", "b c"],
    ...     "label": ["pizza", "sushi", "pizza"]
    ... })
    >>> feature_columns = ["feature1", "feature2"]
    >>> label_column = "label"
    >>> modified_df, vocab = text_to_numeric(df, feature_columns, label_column)
    >>> print(modified_df.to_string())
      feature1 feature2  label  nb_prob_pizza_feature1  nb_prob_sushi_feature1  nb_prob_pizza_feature2  nb_prob_sushi_feature2
    0      a c        b  pizza                0.654987                0.345013                0.919294                0.080706
    1        a      a c  sushi                0.240356                0.759644                0.136594                0.863406
    2      b c      b c  pizza                0.883636                0.116364                0.850642                0.149358
    """
    modified_df = df.copy()
    vocab = []
    for feature in feature_columns:
        vocab = extract_vocab(df, feature)

        data_pairs = list(zip(df[feature].astype(str).tolist(), df[label_column].tolist()))
        X, t, label_mapping = make_bow(data_pairs, vocab)

        pi, theta = naive_bayes_map(X, t, len(label_mapping))
        modified_df = integrate_nb_predictions(modified_df, X, pi, theta, label_mapping, feature)

    return modified_df, vocab

if __name__ == "__main__":
    file_name = "cleaned_data_combined.csv"
    data_path = Path.cwd().parent / file_name
    random_state = 42
    # Read the CSV file from the parent directory.

    print("Data file:", data_path)
    df = pd.read_csv(data_path)


    feature_columns = [
        # "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
        # "Q2: How many ingredients would you expect this food item to contain?",
        "Q3: In what setting would you expect this food to be served? Please check all that apply",
        # "Q4: How much would you expect to pay for one serving of this food item?",
        "Q5: What movie do you think of when thinking of this food item?",
        "Q6: What drink would you pair with this food item?",
        # "Q7: When you think about this food item, who does it remind you of?",
        # "Q8: How much hot sauce would you add to this food item?"
    ]
    label_column = "Label"


    # Merge all text cols.
    df_selected = select_features(df, feature_columns, label_column)
    # Shuffle the dataset.
    df_selected = df_selected.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Extract vocabulary from the combined text field.
    vocab = extract_vocab(df_selected, "combined_text")
    print(f"Combined vocabulary size: {len(vocab)}")

    data_pairs = list(zip(df_selected["combined_text"].tolist(), df_selected[label_column].tolist()))

    # Build bow matrix and label mapping.
    X, t, label_mapping = make_bow(data_pairs, vocab)
    print("Label mapping:", label_mapping)

    n_train = int(0.8 * X.shape[0])
    X_train, t_train = X[:n_train], t[:n_train]
    X_test, t_test = X[n_train:], t[n_train:]

    n_labels = len(label_mapping)
    pi, theta = naive_bayes_map(X_train, t_train, n_labels)

    train_preds = make_prediction(X_train, pi, theta)
    test_preds = make_prediction(X_test, pi, theta)

    print(f"Naive Bayes MAP Train Accuracy: {accuracy(train_preds, t_train):.4f}")
    print(f"Naive Bayes MAP Test Accuracy: {accuracy(test_preds, t_test):.4f}")

    # # Convert numeric predictions back to label strings for display.
    # inv_label_mapping = {v: k for k, v in label_mapping.items()}
    # test_pred_labels = [inv_label_mapping[p] for p in test_preds]
    # print("Sample Test Predictions:", test_pred_labels[:10])

    # from sklearn.linear_model import LogisticRegression
    #
    # lr = LogisticRegression(max_iter=1000)
    #
    # lr.fit(X_train, t_train)
    #
    # train_acc = lr.score(X_train, t_train)
    # val_acc = lr.score(X_test, t_test)
    #
    # print("LR Train Acc:", train_acc)
    # print("LR Valid Acc:", val_acc)



