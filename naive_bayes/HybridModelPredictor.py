"""A hybrid model classifier with limited import."""


import pickle
import re
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd


class HybridModelPredictor:
    """
    A hybrid model class that predicts combinations of numeric, text, and categorical feature processing.
    No training is performed.
    All trained parameters should be loaded from a pickle file.
    """
    def __init__(self, num_cols, cate_cols, text_cols, prob_cols, last_model):
        self.num_cols = num_cols
        self.cate_cols = cate_cols
        self.text_cols = text_cols
        self.prob_cols = prob_cols
        self.label_mapping = None

        self.model_params = {
            "num_cols": {},
            "bayes_cols": {},
            "text_cols": {},
            "label_mapping": {},
            "logreg": {}
        }
        self.last_model = last_model

    @staticmethod
    def extract_vocab(series):
        """
        Extract sorted vocabulary from text data.
        """
        vocab_set = set()
        for text in series.dropna():
            words = re.findall(r"\b\w+\b", str(text).lower())
            vocab_set.update(words)
        return sorted(vocab_set)

    @staticmethod
    def extract_categories(series) -> List[str]:
        """
        Extract unique categories from categorical data.
        """
        option_set = set()
        for options in series.dropna():
            # Split the string by commas and add each option to the set after stripping whitespace
            for option in str(options).split(','):
                option_set.add(option.strip())
        return list(option_set)

    @staticmethod
    def categories_to_binary_matrix(series, categories) -> np.ndarray:
        """
        Convert categorical data into binary features.
        """
        cat_to_index = {cat: idx for idx, cat in enumerate(categories)}
        X_bin = np.zeros((len(series), len(categories)), dtype=int)
        # Iterate through each text entry to populate the binary feature matrix
        for i, options in enumerate(series.fillna('')):
            for option in str(options).split(','):
                option = option.strip()
                if option in cat_to_index:
                    X_bin[i, cat_to_index[option]] = 1
        return X_bin

    @staticmethod
    def text_to_binary_matrix(series, vocab):
        """
        Convert text data into binary features.
        """
        word_to_index = {word: idx for idx, word in enumerate(vocab)}
        X_bin = np.zeros((len(series), len(vocab)), dtype=int)
        for i, text in enumerate(series.fillna("")):
            words = re.findall(r"\b\w+\b", str(text).lower())
            for w in words:
                if w in word_to_index:
                    X_bin[i, word_to_index[w]] = 1
        return X_bin

    @staticmethod
    def parse_number_extraction(response):
        """
        Extract numerical values from text responses.
        """
        if pd.isna(response):
            return np.nan
        response = str(response).strip().lower()
        nums = re.findall(r"\d+\.\d+|\d+", response)
        nums = [float(n) for n in nums]
        if len(nums) == 1:
            return nums[0]
        elif len(nums) > 1:
            return sum(nums) / len(nums)
        return np.nan

    @staticmethod
    def naive_bayes_map(X, t, n_labels=3, a_class = 1.0, a_feat=1.0, b_feat=1.0):
        """
        Compute class probabilities using a trained Bernoulli Naive Bayes model.

        Parameters:
            X (np.ndarray): Binary feature matrix [N, V], where N is the number of samples
                            and V is the number of features.
            pi (np.ndarray): Class prior probabilities [C], where C is the number of classes.
            theta (np.ndarray): Feature likelihood matrix [V, C], with values representing
                                the probability of each feature given the class.

        Returns:
            np.ndarray: Class probability matrix [N, C], where each row sums to 1.
        """
        N, V = X.shape
        K = n_labels
        Y = np.zeros((N, K))
        Y[np.arange(N), t] = 1

        Nk = Y.sum(axis=0)
        pi = (Nk + a_class) / (N + K * a_class)
        theta = (X.T @ Y + a_feat) / (Nk + a_feat + b_feat)
        return pi, theta

    @staticmethod
    def compute_nb_probabilities(X, pi, theta):
        """
        Compute Naive Bayes probabilities.

        Parameters

        """
        log_probs = X @ np.log(theta) + (1 - X) @ np.log(1 - theta) + np.log(pi)
        log_probs -= np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs)
        return probs / probs.sum(axis=1, keepdims=True)

    @staticmethod
    def compute_logistic_probabilities(X, w, b):
        """
        Compute class probabilities using logistic regression parameters.

        Parameters:
            X (np.ndarray): Feature matrix [N, D].
            w (np.ndarray): Logistic regression weights [D] for binary or [C, D] for multiclass.
            b (np.ndarray or float): Bias terms [1] for binary or [C] for multiclass.

        Returns:
            np.ndarray: Probability matrix [N, C] for multiclass, or [N] for binary classification.
        """
        if w.ndim == 1:
            # 1D cases
            logits = X @ w + b
        else:
            logits = X @ w.T + b
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probabilities


    def create_x(self, df):
        """
        Construct a combined feature matrix from the input DataFrame using numeric, text, and categorical features.

        Parameters:
            df (pd.DataFrame): DataFrame containing all relevant feature columns.
                Note: feature columns must include cols in self.num_cols, self.cate_cols or self.text_cols.

        Returns:
            np.ndarray: Combined feature matrix [N, total_features].
        """
        X_parts = []

        for col in self.num_cols:
            val = df[col].apply(self.parse_number_extraction)
            impute_val = self.model_params["num_cols"][col]["impute_value"]
            val.fillna(impute_val, inplace=True)
            X_parts.append(val.to_numpy().reshape(-1, 1))


        for col in self.text_cols:
            vocab = self.model_params["bayes_cols"][col]["vocab"]
            X_bin = self.text_to_binary_matrix(df[col], vocab)
            if col in self.prob_cols:
                pi = np.array(self.model_params["bayes_cols"][col]["pi"])
                theta = np.array(self.model_params["bayes_cols"][col]["theta"])
                X_bin = self.compute_nb_probabilities(X_bin, pi, theta)
            X_parts.append(X_bin)

        for col in self.cate_cols:
            vocab = self.model_params["text_cols"][col]["vocab"]
            X_bin = self.categories_to_binary_matrix(df[col], vocab)
            if col in self.prob_cols:
                w = np.array(self.model_params["text_cols"][col]["w"])
                b = np.array(self.model_params["text_cols"][col]["b"])
                X_bin = self.compute_logistic_probabilities(X_bin, w, b)
            X_parts.append(X_bin)
        return np.hstack(X_parts)


    def predict(self, df):
        """
        Predict class labels for the given DataFrame using the trained hybrid model.

        Parameters:
            df (pd.DataFrame): DataFrame containing features for prediction.

        Returns:
            np.ndarray: Predicted class labels [N].
        """
        X = self.create_x(df)
        return self.last_model.predict(X)

    def predict_class(self, df):
        numeric_preds = self.predict(df)
        reverse_mapping = {v: k for k, v in self.label_mapping.items()}
        vectorized_map = np.vectorize(reverse_mapping.get)
        return vectorized_map(numeric_preds)



    def load(self, path):
        with open(path, 'rb') as f:
            self.model_params = pickle.load(f)
            self.label_mapping = self.model_params["label_mapping"]


class SoftmaxModel:
    """
    A simple softmax-based classifier that does not need to be fit.
    You must provide or load its parameters (weights, bias, class mapping) before prediction.
    """

    def __init__(self, W, b):
        self.W = W
        self.b = b

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def predict(self, X):
        if self.W is None or self.b is None:
            raise ValueError("Model parameters W and b must be set or loaded before predicting.")
        logits = X @ self.W.T + self.b
        probs = self.softmax(logits)
        numeric_preds = np.argmax(probs, axis=1)
        return numeric_preds

def pred(file_path, param_path, feature_config_path):
    df = pd.read_csv(file_path)
    with open(param_path, 'rb') as f:
        model_params = pickle.load(f)
    with open(feature_config_path, 'rb') as f:
        best_features = pickle.load(f)
    W =  model_params["last_model"]["coef_"]
    b = model_params["last_model"]["intercept_"]
    last_model = SoftmaxModel(W, b)
    hmodel = HybridModelPredictor(
        num_cols=best_features["final_num_cols"],
        text_cols=best_features["final_text_cols"],
        cate_cols=best_features["final_cate_cols"],
        prob_cols=best_features["final_prob_cols"],
        last_model=last_model
                                  )
    hmodel.load(param_path)
    return hmodel.predict_class(df)

if __name__ == "__main__":
    file_name = "cleaned_data_combined.csv"
    data_path = Path.cwd().parent / file_name
    y = pred(data_path, "model_params.pkl", "best_config.pkl")
    df = pd.read_csv(data_path)

    t = df["Label"]

    accuracy = np.mean(y == t)

    # Print accuracy
    print(f"Train Accuracy: {accuracy * 100:.2f}%")




