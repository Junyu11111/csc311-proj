import itertools
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.ma.core import indices
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV

from naive_bayes.HybridModelPredictor import HybridModelPredictor


class HybridModel(HybridModelPredictor):
    """
    A hybrid model class that combines numeric, text, and categorical feature processing.
    The final estimator (last_model) is typically a classifier such as Softmax.
    """
    def __init__(self, num_cols, cate_cols, text_cols, prob_cols, label_col, last_model=LogisticRegression(max_iter=10000)):
        """
        Initialize the HybridModel.

        Parameters:
            num_cols (list[str]): Columns treated as numeric.
            cate_cols (list[str]): Columns treated as categorical.
            text_cols (list[str]): Columns treated as text.
            prob_cols (list[str]): Columns for special "processed" transformations.
            label_col (str): Name of the column containing labels.
            last_model (estimator): Final model estimator (default: LogisticRegression).
        """
        super().__init__(num_cols, cate_cols, text_cols, prob_cols, last_model)
        self.label_col = label_col


    def create_t(self, df: pd.DataFrame):
        """
        Create an integer label array from the DataFrame's label column.

        Parameters:
            df (pd.DataFrame): DataFrame containing the label column.

        Returns:
            np.ndarray: Integer label array of shape [N].
        """
        labels = df[self.label_col].fillna("").astype(str)
        N = len(df)
        unique_labels = sorted({label.strip() for label in labels})
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        t = np.zeros((N), dtype=int)
        for i, label in enumerate(labels):
            label_clean = label.strip()
            if label_clean in self.label_mapping:
                t[i] = self.label_mapping[label_clean]
        return t

    def train(self, df):
        """
        Train the hybrid model on a DataFrame of features and labels.

        Parameters:
            df (pd.DataFrame): Training data including label_col.

        Notes:
            - Numeric columns are imputed with their mean.
            - Text columns are optionally processed with a Bayesian search for NB hyperparameters.
            - Categorical columns are optionally processed via logistic regression probabilities.
            - The final estimator is then trained on the combined feature matrix.
        """
        t = self.create_t(df)
        n_labels = len(np.unique(t))
        X_parts = []

        for col in self.num_cols:
            val = df[col].apply(self.parse_number_extraction)
            impute_val = val.mean()
            self.model_params["num_cols"][col] = {"impute_value": impute_val}
            val.fillna(impute_val, inplace=True)
            X_parts.append(val.to_numpy().reshape(-1, 1))

        for col in self.text_cols:
            vocab = self.extract_vocab(df[col])
            params = {"vocab": vocab}
            X_bin = self.text_to_binary_matrix(df[col], vocab)
            if col in self.prob_cols:
                # Apply BayesSearchCV to search for hyperparameters a, b.
                optimizer = BayesSearchCV(
                    estimator=NBParameterEstimator(),
                    search_spaces={
                        'a_feat': (1e-4, 1e2, 'log-uniform'),
                        'b_feat': (1e-4, 1e2, 'log-uniform')
                    },
                    n_iter=10,
                    cv=5,
                    scoring='accuracy'
                )
                optimizer.fit(X_bin, t)
                best_params = optimizer.best_params_
                a, b = best_params['a_feat'], best_params['b_feat']
                pi, theta = self.naive_bayes_map(X_bin, t, n_labels=n_labels, a_feat=a, b_feat=b)
                params.update({"pi": pi, "theta": theta})
                X_bin = self.compute_nb_probabilities(X_bin, pi, theta)
            X_parts.append(X_bin)
            self.model_params["bayes_cols"][col] = params

        for col in self.cate_cols:
            vocab = self.extract_categories(df[col])
            params = {"vocab": vocab}
            X_bin = self.categories_to_binary_matrix(df[col], vocab)
            if col in self.prob_cols:
                lr = LogisticRegression(max_iter=10000)
                lr.fit(X_bin, t)
                params.update({"w": lr.coef_, "b": lr.intercept_})
                X_bin = self.compute_logistic_probabilities(X_bin, lr.coef_, lr.intercept_)
            self.model_params["text_cols"][col] = params
            X_parts.append(X_bin)

        self.last_model.fit(np.hstack(X_parts), t)
        self.model_params["last_model"] = {}
        self.model_params["last_model"]['coef_'] = self.last_model.coef_
        self.model_params["last_model"]['intercept_'] = self.last_model.intercept_
        self.model_params["label_mapping"] = self.label_mapping


    def prep_and_store_X(self, df: pd.DataFrame):
        """
        Precompute raw and processed representations for each column and store them in self.x_dict.

        Parameters:
            df (pd.DataFrame): DataFrame containing all feature columns + label_col.
        """
        t = self.create_t(df)
        n_labels = len(np.unique(t))
        self.x_dict = {"t":t}

        for col in self.num_cols:
            val = df[col].apply(self.parse_number_extraction)
            impute_val = val.mean()
            if pd.isna(impute_val):
                impute_val = 0.0
            self.model_params["num_cols"][col] = {"impute_value": impute_val}
            val.fillna(impute_val, inplace=True)
            self.x_dict[col + "_num"] = val.to_numpy().reshape(-1, 1)


        for col in self.text_cols:
            vocab = self.extract_vocab(df[col])
            params = {"vocab": vocab}
            X_bin = self.text_to_binary_matrix(df[col], vocab)
            self.x_dict[col + "_text"] =  X_bin
            optimizer = BayesSearchCV(
                estimator=NBParameterEstimator(),
                search_spaces={
                    'a_feat': (0.1, 10.0, 'log-uniform'),
                    'b_feat': (0.1, 10.0, 'log-uniform')
                },
                n_iter=10,
                cv=3,
                scoring='accuracy'
            )
            optimizer.fit(X_bin, t)
            best_params = optimizer.best_params_
            a, b = best_params['a_feat'], best_params['b_feat']
            pi, theta = self.naive_bayes_map(X_bin, t, n_labels=n_labels, a_feat=a, b_feat=b)
            params.update({"pi": pi, "theta": theta})
            self.x_dict[col + "_text" + "_prob"] = self.compute_nb_probabilities(X_bin, pi, theta)
            self.model_params["bayes_cols"][col] = params

        for col in self.cate_cols:
            vocab = self.extract_vocab(df[col])
            params = {"vocab": vocab}
            X_bin = self.text_to_binary_matrix(df[col], vocab)
            self.x_dict[col + "_cate"] = X_bin
            lr = LogisticRegression(max_iter=10000)
            lr.fit(X_bin, t)
            params.update({"w": lr.coef_, "b": lr.intercept_})
            self.x_dict[col + "_cate" + "_prob"] = self.compute_logistic_probabilities(X_bin, lr.coef_, lr.intercept_)
            self.model_params["text_cols"][col] = params

    def optimize_feature_selection(self, df, cv=5, output_config_file="best_config.pkl"):
        """
        Enumerate 'none', 'raw', and 'prob' combinations for each column, then use cross-validation
        to find the best feature configuration.

        The list `representation_types = ["none", "raw", "prob"]` works as follows:
            - "none": Exclude this column from the feature matrix (no features used).
            - "raw": Include this column's 'raw' representation (e.g., numeric imputed data,
                     bag-of-words for text, or one-hot for categorical).
            - "prob": Include this column's 'processed' representation (e.g., Naive Bayes
                      probabilities for text or logistic probabilities for categorical).

        Returns:
            (dict, float): Tuple of (best configuration dictionary, best CV score).
        """
        self.prep_and_store_X(df)
        columns = [f"{col}_num" for col in self.num_cols] \
                  + [f"{col}_text" for col in self.text_cols] \
                  + [f"{col}_cate" for col in self.cate_cols]
        representation_types = ["none", "raw", "prob"]

        best_score = -np.inf
        best_config = None

        # Itertools.product for each column's representation choice
        import itertools
        c = 0
        for rep_tuple in itertools.product(representation_types, repeat=len(columns)):
            # skip if all "none"
            if all(r == "none" for r in rep_tuple):
                continue
            # build X using the chosen representation for each column
            X_parts = []
            for col, rep_type in zip(columns, rep_tuple):
                if rep_type == "none":
                    # skip
                    continue
                elif rep_type == "raw":
                    X_parts.append(self.x_dict[col])
                elif rep_type == "prob" and (col not in self.num_cols):
                    X_parts.append(self.x_dict[col + "_prob"])
            X_full = np.hstack(X_parts) if X_parts else np.zeros((len(df), 0))

            if X_full.shape[1] == 0:
                continue

            # Evaluate using cross-validation on the entire dataset
            scores = cross_val_score(self.last_model, X_full, self.x_dict['t'], cv=cv, scoring='accuracy')
            avg_score = scores.mean()

            if avg_score > best_score:
                best_score = avg_score
                chosen_config = dict(zip(columns, rep_tuple))
                best_config = chosen_config
                print("new best config:", best_score, best_config)
            if c % 10 == 0:
                print(c)
            c += 1

        # Build final dict describing the chosen columns

        final_num_cols = []
        final_text_cols = []
        final_cate_cols = []
        final_prob_cols = []

        for col in self.num_cols:
            choice = best_config.get(col + "_num", "none")
            if choice == "raw":
                final_num_cols.append(col)
            elif choice == "prob":
                final_num_cols.append(col)
                final_prob_cols.append(col)

        for col in self.text_cols:
            choice = best_config.get(col + "_text", "none")
            if choice == "raw":
                final_text_cols.append(col)
            elif choice == "prob":
                final_text_cols.append(col)
                final_prob_cols.append(col)

        for col in self.cate_cols:
            choice = best_config.get(col + "_cate", "none")
            if choice == "raw":
                final_cate_cols.append(col)
            elif choice == "prob":
                final_cate_cols.append(col)
                final_prob_cols.append(col)

        best_config_dict = {
            "final_num_cols": final_num_cols,
            "final_text_cols": final_text_cols,
            "final_cate_cols": final_cate_cols,
            "final_prob_cols": final_prob_cols,
            "score": best_score
        }
        with open(output_config_file, "wb") as f:
            pickle.dump(best_config_dict, f)

        print(f"Best config saved to {output_config_file}")
        print("Best config dictionary:", best_config_dict)
        return best_config_dict, best_score


    def save(self, path):
        """
        Save the entire HybridModel (including parameters) to a file via pickle.

        Parameters:
            path (str): Filepath to store the pickled model.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.model_params, f)

    def score(self, df):
        """
        Compute the accuracy of the final model on a new DataFrame.

        Parameters:
            df (pd.DataFrame): DataFrame with all necessary columns + label_col.

        Returns:
            float: Accuracy (mean of correct predictions).
        """
        y = self.predict(df)
        t = self.create_t(df)
        return np.mean(y == t)






from sklearn.base import BaseEstimator

class NBParameterEstimator(BaseEstimator):
    """
    A custom estimator that wraps the naive_bayes_map transformation.
    It uses hyperparameters a_feat and b_feat for smoothing.
    """
    def __init__(self, a_feat=1.0, b_feat=1.0):
        self.a_feat = a_feat
        self.b_feat = b_feat


    @staticmethod
    def make_prediction(X, pi, theta):
        log_probs = np.dot(X, np.log(theta)) + np.dot(1 - X, np.log(1 - theta)) + np.log(pi)
        return np.argmax(log_probs, axis=1)

    @staticmethod
    def accuracy(y, t):
        return np.mean(y == t)

    def fit(self, X, y):
        n_labels = len(np.unique(y))
        self.pi_, self.theta_ = HybridModelPredictor.naive_bayes_map(X, y, n_labels=n_labels,
                                                                      a_feat=self.a_feat, b_feat=self.b_feat)
        return self

    def transform(self, X):
        return HybridModelPredictor.compute_nb_probabilities(X, self.pi_, self.theta_)

    def predict(self, X):
        return self.make_prediction(X, self.pi_, self.theta_)

    def score(self, X, y):
        """
        Score the estimator using your custom prediction function and accuracy metric.
        """
        y_pred = self.make_prediction(X, self.pi_, self.theta_)
        return self.accuracy(y_pred, y)


if __name__ == "__main__":
    num_cols = [
        # "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
        # "Q2: How many ingredients would you expect this food item to contain?",
        # "Q3: In what setting would you expect this food to be served? Please check all that apply",
        # "Q4: How much would you expect to pay for one serving of this food item?",
        # "Q5: What movie do you think of when thinking of this food item?",
        # "Q6: What drink would you pair with this food item?",
        # "Q7: When you think about this food item, who does it remind you of?",
        # "Q8: How much hot sauce would you add to this food item?"
    ]
    cate_cols = [
        "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
        # "Q2: How many ingredients would you expect this food item to contain?",
        "Q3: In what setting would you expect this food to be served? Please check all that apply",
        # "Q4: How much would you expect to pay for one serving of this food item?",
        # "Q5: What movie do you think of when thinking of this food item?",
        # "Q6: What drink would you pair with this food item?",
        "Q7: When you think about this food item, who does it remind you of?",
        "Q8: How much hot sauce would you add to this food item?"
    ]
    text_columns = [
        # "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
        "Q2: How many ingredients would you expect this food item to contain?",
        # "Q3: In what setting would you expect this food to be served? Please check all that apply",
        "Q4: How much would you expect to pay for one serving of this food item?",
        "Q5: What movie do you think of when thinking of this food item?",
        "Q6: What drink would you pair with this food item?",
        # "Q7: When you think about this food item, who does it remind you of?",
        # "Q8: How much hot sauce would you add to this food item?"
    ]
    prob_columns = [
        "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
        "Q2: How many ingredients would you expect this food item to contain?",
        "Q3: In what setting would you expect this food to be served? Please check all that apply",
        "Q4: How much would you expect to pay for one serving of this food item?",
        "Q5: What movie do you think of when thinking of this food item?",
        "Q6: What drink would you pair with this food item?",
        "Q7: When you think about this food item, who does it remind you of?",
        "Q8: How much hot sauce would you add to this food item?"
    ]

    file_name = "cleaned_data_combined.csv"
    data_path = Path.cwd().parent / file_name
    df = pd.read_csv(data_path)
    df_shuffled = df.sample(frac=1, random_state=42)  # Shuffle rows (set random_state for reproducibility)
    df_train = df_shuffled.iloc[:int(len(df_shuffled) * 0.8)]  # Take first 80% of rows
    df_test = df_shuffled.iloc[int(len(df_shuffled) * 0.8):]

    hmodel = HybridModel(
        num_cols=num_cols,
        text_cols=text_columns,
        prob_cols=prob_columns,
        cate_cols=cate_cols,
        label_col="Label",
        last_model=LogisticRegression(max_iter=10000)
    )
    hmodel.optimize_feature_selection(df)

    with open("best_config.pkl", 'rb') as f:
        best_features = pickle.load(f)
    hmodel = HybridModel(
        num_cols=best_features["final_num_cols"],
        text_cols=best_features["final_text_cols"],
        cate_cols=best_features["final_cate_cols"],
        prob_cols=best_features["final_prob_cols"],
        label_col="Label",
        last_model=LogisticRegression(max_iter=10000)
    )
    hmodel.train(df)
    hmodel.save("model_params.pkl")





