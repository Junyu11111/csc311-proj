"""Clean up CSV, create training matrix and test Logistic Regression"""
import re
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Union
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

TRAINFILE = "cleaned_data_combined_modified.csv"


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
        words = re.findall(r'\b\w+\b', text.lower())
        vocab_set.update(words)
    return sorted(vocab_set)


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


def create_text_features(df: pd.DataFrame, column: str) -> np.ndarray:
    """
    Converts a text column into a binary feature matrix based on the vocabulary.

    Parameters:
        df (pd.DataFrame): DataFrame containing the text data.
        column (str): The column name to process.

    Returns:
        np.ndarray: A binary matrix where each row represents an observation and each column
                    represents whether the corresponding vocabulary word appears in the text.
    """
    vocab = extract_vocab(df, column)
    word_to_index: Dict[str, int] = {word: idx for idx, word in enumerate(vocab)}
    N = len(df)
    V = len(vocab)
    X_bin = np.zeros((N, V), dtype=int)

    # Iterate through each text entry to populate the binary feature matrix
    for i, text in enumerate(df[column].fillna('')):
        words = re.findall(r'\b\w+\b', text.lower())
        for word in words:
            idx = word_to_index.get(word)
            if idx is not None:
                X_bin[i, idx] = 1
    return X_bin


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


if __name__ == "__main__":
    X, t = create_X_t(TRAINFILE)
    print("Feature matrix shape:", X.shape)
    print("Label matrix shape:", t.shape)

    # Use NumPy to split the data into 80% training and 20% testing sets
    N = X.shape[0]
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)

    train_size = int(0.8 * N)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    t_train, t_test = t[train_indices], t[test_indices]

    print("Training feature matrix shape:", X_train.shape)
    print("Testing feature matrix shape:", X_test.shape)
    print("Training label matrix shape:", t_train.shape)
    print("Testing label matrix shape:", t_test.shape)

    # Initialize the logistic regression model
    test_model = LogisticRegression()

    # Train the model on the training data
    test_model.fit(X_train, t_train)

    # Predict on the test set
    y_pred = test_model.predict(X_test)

    # Evaluate the model's accuracy
    accuracy = accuracy_score(t_test, y_pred)
    print("Test Accuracy:", accuracy)

    # Train model on complete dataset
    final_model = LogisticRegression()
    final_model.fit(X, t)

    # Save weights in weights.txt
    np.savetxt("weights.txt", final_model.coef_, fmt="%.7f")  # Saves with 7 decimal places
    np.savetxt("bias.txt", final_model.intercept_, fmt="%.7f")
