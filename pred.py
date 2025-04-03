import pickle
from collections import Counter
from typing import List
import re
import numpy as np
import pandas as pd


class ModelClassifier:
    def __init__(self, model_params:dict=None):
        if model_params is None:
            self.model_params = {}
        else:
            self.model_params = model_params
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

    def predict(self, X):
        W = np.array(self.model_params["W"])
        b = np.array(self.model_params["b"])
        X = X @ self.model_params["svd_components"].T
        probs = self.compute_logistic_probabilities(X, W, b)
        numeric_preds = np.argmax(probs, axis=1)
        return np.array(self.model_params["classes"])[numeric_preds]

    def load(self, param_filename):
        with open(param_filename, "rb") as f:
            self.model_params = pickle.load(f)



class XGen:
    def __init__(self, num_cols = None, cate_cols=None, text_cols=None, col_params=None, text_method = "binary"):
        self.num_cols = num_cols
        self.cate_cols = cate_cols
        self.text_cols = text_cols
        self.text_method = text_method
        self.col_params = col_params if col_params else {
            'num_cols': {},
            'text_cols': {},
            'cate_cols': {}
        }

    @staticmethod
    def extract_vocab(series, stopwords=None, min_freq=2):
        """
        Extract sorted vocabulary from text data, filtering out stopwords and infrequent words.
        """
        if stopwords is None:
            stopwords = {"the", "and", "is", "a", "an", "of", "to", "in", "for", "on", "with", "at"}
        word_counter = Counter()
        for text in series.dropna():
            words = re.findall(r"\b\w+\b", str(text).lower())
            filtered_words = [word for word in words if word not in stopwords]
            word_counter.update(filtered_words)
        vocab = [word for word, count in word_counter.items() if count >= min_freq]
        return sorted(vocab)

    @staticmethod
    def extract_categories(series) -> List[str]:
        """
        Extract unique categories from categorical data.
        """
        option_set = set()
        for options in series.dropna():
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
        for i, options in enumerate(series.fillna('')):
            for option in str(options).split(','):
                option = option.strip()
                if option in cat_to_index:
                    X_bin[i, cat_to_index[option]] = 1
        return X_bin

    @staticmethod
    def text_to_tfidf_matrix(series, vocab, idf_values = None):
        """
        Convert text data into TF-IDF features.
        """
        word_to_index = {word: idx for idx, word in enumerate(vocab)}
        N = len(series)
        V = len(vocab)
        X_tfidf = np.zeros((N, V), dtype=float)
        for i, text in enumerate(series.fillna("")):
            words = re.findall(r"\b\w+\b", str(text).lower())
            term_counts = Counter(words)
            total_terms = len(words)
            if total_terms == 0:
                continue
            for word, count in term_counts.items():
                if word in word_to_index:
                    tf = count / total_terms
                    idf = idf_values.get(word, 0.0)
                    X_tfidf[i, word_to_index[word]] = tf * idf
        return X_tfidf

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

    def prep_X(self, df):
        """
        Preprocess and extract features from the DataFrame.
        This method is used during training (fit) to build col_params.
        """
        X_parts = []
        # Process numeric columns
        for col in self.num_cols:
            val = df[col].apply(self.parse_number_extraction)
            impute_val = val.mean()
            if pd.isna(impute_val):
                impute_val = 1
            self.col_params["num_cols"][col] = {"impute_value": impute_val}
            val.fillna(impute_val, inplace=True)
            X_parts.append(val.to_numpy().reshape(-1, 1))

        # Process categorical columns
        for col in self.cate_cols:
            vocab = self.extract_categories(df[col])
            self.col_params["cate_cols"][col] = {"vocab": vocab}
            X_bin = self.categories_to_binary_matrix(df[col], vocab)
            X_parts.append(X_bin)

        # Process text columns with TF-IDF
        for col in self.text_cols:
            vocab = self.extract_vocab(df[col])
            # Binary
            if self.text_method == "binary":
                self.col_params["text_cols"][col] = {"vocab": vocab}
                X_bin = self.text_to_binary_matrix(df[col], vocab)
                X_parts.append(X_bin)

            # tfidf
            elif self.text_method == "tfidf":
                # Compute document frequencies for IDF
                doc_freq = Counter()
                total_docs = df.shape[0]
                for text in df[col].dropna():
                    unique_terms = set(re.findall(r"\b\w+\b", text.lower()))
                    for term in unique_terms:
                        if term in vocab:
                            doc_freq[term] += 1
                idf_values = {
                    term: np.log(total_docs / (1 + doc_freq[term]))
                    for term in vocab
                }
                self.col_params["text_cols"][col] = {
                    "vocab": vocab,
                    "idf_values": idf_values
                }
                X_tfidf = self.text_to_tfidf_matrix(df[col], vocab, idf_values)
                X_parts.append(X_tfidf)

        return np.hstack(X_parts)

    def create_X(self, df):
        """
        Use stored parameters (col_params) to convert new DataFrame into feature array.
        """
        X_parts = []
        # Process numeric columns
        for col in self.num_cols:
            val = df[col].apply(self.parse_number_extraction)
            impute_val = self.col_params["num_cols"][col]["impute_value"]
            val.fillna(impute_val, inplace=True)
            X_parts.append(val.to_numpy().reshape(-1, 1))

        # Process categorical columns
        for col in self.cate_cols:
            vocab = self.col_params["cate_cols"][col]["vocab"]
            X_bin = self.categories_to_binary_matrix(df[col], vocab)
            X_parts.append(X_bin)

        # Process text columns using stored vocabulary and IDF values
        for col in self.text_cols:
            vocab = self.col_params["text_cols"][col]["vocab"]
            if self.text_method == "binary":
                X = self.text_to_binary_matrix(df[col], vocab)
            elif self.text_method == "tfidf":
                idf_values = self.col_params["text_cols"][col]["idf_values"]
                X = self.text_to_tfidf_matrix(df[col], vocab, idf_values)
            X_parts.append(X)

        return np.hstack(X_parts)


    def load(self, param_filename):
        with open(param_filename, "rb") as f:
            params = pickle.load(f)
            self.set_params(**params)

    def save(self, param_filename):
        with open(param_filename, "wb") as f:
            pickle.dump(self.get_params(), f)


    # Implement fit and transform for pipeline integration
    def fit(self, X, y=None):
        # X is expected to be a pandas DataFrame.
        self.prep_X(X)  # Build and store parameters
        return self

    def transform(self, X):
        # Use stored parameters to create feature matrix
        return self.create_X(X)

    def get_params(self, deep=True):

        return {
            "num_cols": self.num_cols,
            "cate_cols": self.cate_cols,
            "text_cols": self.text_cols,
            "col_params": self.col_params,
            "text_method": self.text_method,
        }

    def set_params(self, **params):

        for key, value in params.items():
            setattr(self, key, value)
        return self


def pipeline(df, xgen:XGen, classifier:ModelClassifier):
    X = xgen.transform(df)
    return classifier.predict(X)

def predict_df(df, xgen_param_filename = "xgen_params.pkl", classifier_param_filename = "classifier_params.pkl"):
    xgen = XGen()
    xgen.load(xgen_param_filename)
    classifier = ModelClassifier()
    classifier.load(classifier_param_filename)
    return pipeline(df, xgen, classifier)


def predict_all(file_name, xgen_param_filename = "xgen_params.pkl", classifier_param_filename = "classifier_params.pkl"):
    df = pd.read_csv(file_name)
    return predict_df(df, xgen_param_filename, classifier_param_filename)

if __name__ == "__main__":
    filename = "cleaned_data_combined_modified.csv"
    df = pd.read_csv(filename)
    t = df["Label"]  # Actual classes
    y = predict_all(filename)  # Predicted classes

    # Compute accuracy
    accuracy = np.mean(y == t)

    # Print accuracy
    print(f"Train Accuracy: {accuracy * 100:.2f}%")

