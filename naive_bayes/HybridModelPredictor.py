import pickle
import re
from typing import List

import numpy as np
import pandas as pd



class HybridModelPredictor:
    def __init__(self, num_cols, cate_cols, text_cols, prepare_cols, last_model):
        self.num_cols = num_cols
        self.cate_cols = cate_cols
        self.text_cols = text_cols
        self.prepare_cols = prepare_cols

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
        vocab_set = set()
        for text in series.dropna():
            words = re.findall(r"\b\w+\b", str(text).lower())
            vocab_set.update(words)
        return sorted(vocab_set)

    @staticmethod
    def extract_categories(series) -> List[str]:
        option_set = set()
        for options in series.dropna():
            # Split the string by commas and add each option to the set after stripping whitespace
            for option in str(options).split(','):
                option_set.add(option.strip())
        return list(option_set)

    @staticmethod
    def categories_to_binary_matrix(series, categories) -> np.ndarray:
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
    def naive_bayes_map(X, t, n_labels=3, a_feat=1.0, b_feat=1.0):
        N, V = X.shape
        K = n_labels
        Y = np.zeros((N, K))
        Y[np.arange(N), t] = 1

        Nk = Y.sum(axis=0)
        pi = (Nk + 1.0) / (N + K * 1.0)
        theta = (X.T @ Y + a_feat) / (Nk + a_feat + b_feat)
        return pi, theta

    @staticmethod
    def compute_nb_probabilities(X, pi, theta):
        log_probs = X @ np.log(theta) + (1 - X) @ np.log(1 - theta) + np.log(pi)
        log_probs -= np.max(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs)
        return probs / probs.sum(axis=1, keepdims=True)

    @staticmethod
    def compute_logistic_probabilities(X, w, b):
        # If w is 1D (binary), use the original calculation.
        if w.ndim == 1:
            logits = X @ w + b
        else:
            # For multiclass, compute X @ w.T so that logits has shape (n_samples, n_classes)
            logits = X @ w.T + b
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probabilities


    def create_x(self, df):
        X_parts = []

        for col in self.num_cols:
            val = df[col].apply(self.parse_number_extraction)
            impute_val = self.model_params["num_cols"][col]["impute_value"]
            val.fillna(impute_val, inplace=True)
            X_parts.append(val.to_numpy().reshape(-1, 1))


        for col in self.text_cols:
            vocab = self.model_params["bayes_cols"][col]["vocab"]
            X_bin = self.text_to_binary_matrix(df[col], vocab)
            if col in self.prepare_cols:
                pi = np.array(self.model_params["bayes_cols"][col]["pi"])
                theta = np.array(self.model_params["bayes_cols"][col]["theta"])
                X_bin = self.compute_nb_probabilities(X_bin, pi, theta)
            X_parts.append(X_bin)

        for col in self.cate_cols:
            vocab = self.model_params["text_cols"][col]["vocab"]
            X_bin = self.categories_to_binary_matrix(df[col], vocab)
            if col in self.prepare_cols:
                w = np.array(self.model_params["text_cols"][col]["w"])
                b = np.array(self.model_params["text_cols"][col]["b"])
                X_bin = self.compute_logistic_probabilities(X_bin, w, b)
            X_parts.append(X_bin)
        return np.hstack(X_parts)


    def predict(self, df):
        X = self.create_x(df)
        return self.last_model.predict(X)


    def load(self, path):
        with open(path, 'rb') as f:
            self.model_params = pickle.load(f)
