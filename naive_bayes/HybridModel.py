import itertools
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score, KFold
from skopt import BayesSearchCV

from naive_bayes.HybridModelPredictor import HybridModelPredictor


class HybridModel(HybridModelPredictor):
    """
    A hybrid model class that combines numeric, text, and categorical feature processing.
    The final estimator (last_model) is typically a classifier such as Softmax.
    """
    def __init__(self, num_cols, cate_cols, text_cols, prob_dict, label_col, random_state=42):
        """
        Initialize the HybridModel.

        Parameters:
            num_cols (list[str]): Columns treated as numeric.
            cate_cols (list[str]): Columns treated as categorical.
            text_cols (list[str]): Columns treated as text.
            prob_dict (dict): Columns for special "processed" transformations.
            label_col (str): Name of the column containing labels.
            last_model (estimator): Final model estimator (default: LogisticRegression).
        """
        super().__init__(num_cols, cate_cols, text_cols, prob_dict)
        self.label_col = label_col
        self.random_state = random_state


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

    def train(self, df, tune_hyperparams: bool):
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

        X = self.prep_X(df, tune_hyperparams=tune_hyperparams)

        params = {}

        if tune_hyperparams:
            lr = LogisticRegression(max_iter=10000, random_state=self.random_state)

            optimizer = BayesSearchCV(
                estimator=lr,
                search_spaces={
                    'C': (0.01, 100.0, 'log-uniform'),
                },
                n_iter=10,
                cv=3,
                scoring='accuracy',
                random_state=self.random_state
            )
            optimizer.fit(X, t)
            best_params = optimizer.best_params_
        else:
            best_params = self.model_params["last_model"]["hyperparams"]
        best_lr = LogisticRegression(
            C=best_params['C'],
            max_iter=10000,
            random_state=self.random_state
        )
        best_lr.fit(X, t)
        params.update({
            "w": best_lr.coef_,
            "b": best_lr.intercept_,
            "hyperparams": best_params
        })
        self.model_params["last_model"] = params
        self.model_params["label_mapping"] = self.label_mapping
        return self.model_params


    def prep_X(self, df: pd.DataFrame, tune_hyperparams=True):
        self.prep_X_dict(df, prep_all_prob=False, tune_hyperparams=tune_hyperparams)
        X_parts = []
        columns = [f"{col}_num" for col in self.num_cols] \
                  + [f"{col}_text" for col in self.text_cols] \
                  + [f"{col}_cate" for col in self.cate_cols]
        for col in columns:
            X_parts.append(self.X_dict[col])
            if f"{col}_prob" in self.X_dict:
                # append converted probabilities to X
                X_parts.append(self.X_dict[f"{col}_prob"])
        return np.hstack(X_parts)




    def prep_X_dict(self, df: pd.DataFrame, prep_all_prob=True, tune_hyperparams=True):
        """
        Precompute raw and processed representations for each column and store them in self.x_dict.

        Parameters:
            df (pd.DataFrame): DataFrame containing all feature columns + label_col.
        """

        t = self.create_t(df)
        n_labels = len(np.unique(t))
        self.X_dict = {"t":t}

        for col in self.num_cols:
            val = df[col].apply(self.parse_number_extraction)
            impute_val = val.mean()
            if pd.isna(impute_val):
                impute_val = 0.0
            self.model_params["num_cols"][col] = {"impute_value": impute_val}
            val.fillna(impute_val, inplace=True)
            self.X_dict[col + "_num"] = val.to_numpy().reshape(-1, 1)


        for col in self.text_cols:
            vocab = self.extract_vocab(df[col])
            params = {"vocab": vocab}
            X_bin = self.text_to_binary_matrix(df[col], vocab)
            self.X_dict[col + "_text"] =  X_bin
            if prep_all_prob or "text" in self.prob_dict[col]:
                # convert if probability convertion is specified
                if tune_hyperparams:
                    optimizer = BayesSearchCV(
                        estimator=NBParameterEstimator(),
                        search_spaces={
                            'a_feat': (0.1, 10.0, 'log-uniform'),
                            'b_feat': (0.1, 10.0, 'log-uniform')
                        },
                        n_iter=10,
                        cv=5,
                        scoring='accuracy',
                        random_state=self.random_state,
                    )
                    optimizer.fit(X_bin, t)
                    best_params = optimizer.best_params_
                else:
                    best_params = self.model_params["text_cols"][col]["hyperparams"]
                a, b = best_params['a_feat'], best_params['b_feat']
                pi, theta = self.naive_bayes_map(X_bin, t, n_labels=n_labels, a_feat=a, b_feat=b)
                params.update({"pi": pi, "theta": theta, "hyperparams": best_params})
                self.X_dict[col + "_text" + "_prob"] = self.compute_nb_probabilities(X_bin, pi, theta)
            self.model_params["text_cols"][col] = params

        for col in self.cate_cols:
            vocab = self.extract_categories(df[col])
            params = {"vocab": vocab}
            X_bin = self.categories_to_binary_matrix(df[col], vocab)
            self.X_dict[col + "_cate"] = X_bin
            if prep_all_prob or "cate" in self.prob_dict[col]:
                # convert if probability convertion is specified
                if tune_hyperparams:
                    lr = LogisticRegression(max_iter=10000, random_state=self.random_state)
                    optimizer = BayesSearchCV(
                        estimator=lr,
                        search_spaces={
                            'C': (0.01, 100.0, 'log-uniform'),
                        },
                        n_iter=10,
                        cv=5,
                        scoring='accuracy',
                        random_state=self.random_state
                    )
                    optimizer.fit(X_bin, t)
                    best_params = optimizer.best_params_
                else:
                    best_params = self.model_params["cate_cols"][col]["hyperparams"]

                best_lr = LogisticRegression(
                    C=best_params['C'],
                    max_iter=10000,
                    random_state=self.random_state
                )
                best_lr.fit(X_bin, t)
                params.update({
                    "w": best_lr.coef_,
                    "b": best_lr.intercept_,
                    "hyperparams": best_params
                })
                self.X_dict[col + "_cate" + "_prob"] = self.compute_logistic_probabilities(
                    X_bin, best_lr.coef_, best_lr.intercept_
                )

            self.model_params["cate_cols"][col] = params

    def optimize_feature_selection(self, df, cv=5, output_config_file="best_config.pkl", prep_all_prob=True):
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
        self.prep_X_dict(df, prep_all_prob, tune_hyperparams=True)
        columns = []
        available_reps = {}  # Maps column name to available representation types

        # For numerical columns
        for col in self.num_cols:
            col_name = f"{col}_num"
            columns.append(col_name)
            available_reps[col_name] = ["none", "raw"]
            if f"{col_name}_prob" in self.X_dict:
                available_reps[col_name].append("prob")

        # For text columns
        for col in self.text_cols:
            col_name = f"{col}_text"
            columns.append(col_name)
            available_reps[col_name] = ["none", "raw"]
            if f"{col_name}_prob" in self.X_dict:
                available_reps[col_name].append("prob")

        # For categorical columns
        for col in self.cate_cols:
            col_name = f"{col}_cate"
            columns.append(col_name)
            available_reps[col_name] = ["none", "raw"]
            if f"{col_name}_prob" in self.X_dict:
                available_reps[col_name].append("prob")

        best_score = -np.inf
        best_config = None

        # Generate product of only available representation types for each column
        from itertools import product

        c = 0
        # Create list of available representation options for each column
        rep_options = [available_reps[col] for col in columns]

        for rep_tuple in product(*rep_options):
            if all(r == "none" for r in rep_tuple):
                continue

            X_parts = []
            for col, rep_type in zip(columns, rep_tuple):
                if rep_type == "none":
                    continue
                elif rep_type == "raw":
                    X_parts.append(self.X_dict[col])
                elif rep_type == "prob":
                    X_parts.append(self.X_dict[f"{col}_prob"])

            X_full = np.hstack(X_parts) if X_parts else np.zeros((len(df), 0))

            if X_full.shape[1] == 0:
                continue
            scores = cross_val_score(LogisticRegression(), X_full, self.X_dict['t'], cv=cv, scoring='accuracy')
            avg_score = scores.mean()

            if avg_score > best_score:
                best_score = avg_score
                chosen_config = dict(zip(columns, rep_tuple))
                best_config = chosen_config
                print("new best config:", best_score, best_config)

            if c % 10 == 0:
                print(f"Progress: {c}")
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
        self.num_cols = final_num_cols

        for col in self.text_cols:
            choice = best_config.get(col + "_text", "none")
            if choice == "raw":
                final_text_cols.append(col)
            elif choice == "prob":
                final_text_cols.append(col)
                final_prob_cols.append(col)
        self.text_cols = final_text_cols

        for col in self.cate_cols:
            choice = best_config.get(col + "_cate", "none")
            if choice == "raw":
                final_cate_cols.append(col)
            elif choice == "prob":
                final_cate_cols.append(col)
                final_prob_cols.append(col)
        self.cate_cols = final_cate_cols
        self.prob_cols = final_prob_cols

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
            path (str): Filepath to store the model params.
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


def cross_validation_test(df, hmodel, n_splits=5):
    """
    Perform cross validation on the given DataFrame using the specified model.

    Parameters:
    - df (pd.DataFrame): The dataset to perform cross validation on.
    - hmodel: An instance of the model class with train and score methods.
    - n_splits (int): The number of folds for cross validation.

    Returns:
    - scores (list): List of scores from each fold.
    - avg_score (float): The average score across all folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df), start=1):
        # Split the DataFrame into training and testing folds
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]


        # Train the model on the training data with hyperparameter tuning enabled
        hmodel.train(df_train, tune_hyperparams=True)

        # Evaluate the model on the testing data
        score = hmodel.score(df_test)
        scores.append(score)
    avg_score = sum(scores) / len(scores)

    return scores, avg_score


def feature_selection_best_combination(df,
                                       candidate_numeric_features,
                                       candidate_text_features,
                                       candidate_categorical_features,
                                       prob_dict,
                                       n_splits=5,
                                       output_config_file="best_config.pkl",
                                       loop_numeric=True,
                                       loop_text=True,
                                       loop_categorical=True):

    best_result = {}
    best_avg_score = float("-inf")  # Assuming higher score is better
    c = 0

    # Generate combinations based on the provided flags.
    if loop_numeric:
        numeric_combinations = list(itertools.chain.from_iterable(
            itertools.combinations(candidate_numeric_features, r)
            for r in range(0, len(candidate_numeric_features) + 1))
        )
    else:
        numeric_combinations = [tuple(candidate_numeric_features)]

    if loop_text:
        text_combinations = list(itertools.chain.from_iterable(
            itertools.combinations(candidate_text_features, r)
            for r in range(0, len(candidate_text_features) + 1))
        )
    else:
        text_combinations = [tuple(candidate_text_features)]


    if loop_categorical:
        categorical_combinations = list(itertools.chain.from_iterable(
            itertools.combinations(candidate_categorical_features, r)
            for r in range(0, len(candidate_categorical_features) + 1))
        )
    else:
        categorical_combinations = [tuple(candidate_categorical_features)]

    # Iterate over every combination from each feature group.
    for numeric_subset in numeric_combinations:
        for text_subset in text_combinations:
            for categorical_subset in categorical_combinations:
                # Skip if no features are selected from all groups.
                if not (text_subset or categorical_subset):
                    continue

                # Convert tuples to lists for the model.
                numeric_features = list(numeric_subset)
                text_features = list(text_subset)
                categorical_features = list(categorical_subset)
                print("Testing combination:")
                print("  Numeric:", numeric_features)
                print("  Text:", text_features)
                print("  Categorical:", categorical_features)

                h_model = HybridModel(
                    num_cols=numeric_features,
                    text_cols=text_features,
                    prob_dict=prob_dict,
                    cate_cols=categorical_features,
                    label_col="Label",
                )

                print(cross_validation_test(df, h_model, n_splits=5))
                scores, avg_score = cross_validation_test(df, h_model, n_splits=n_splits)
                print(avg_score)

                # Store the combination if it's the best so far.
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    best_result = {
                        "numeric_features": numeric_features,
                        "text_features": text_features,
                        "categorical_features": categorical_features,
                        "prob_dict": prob_dict,
                        "scores": scores,
                        "avg_score": avg_score
                    }
                    print("New best combination found with average score:", best_avg_score)

                c += 1
                if c % 10 == 0:
                    print("Tested combinations:", c)
    # Save the best configuration to file.
    if best_result is not None:
        with open(output_config_file, "wb") as f:
            pickle.dump(best_result, f)
        print("Best configuration saved to", output_config_file)
    else:
        print("No valid configuration found.")

    return best_result



if __name__ == "__main__":

    features = [
        "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
        "Q2: How many ingredients would you expect this food item to contain?",
        "Q3: In what setting would you expect this food to be served? Please check all that apply",
        "Q4: How much would you expect to pay for one serving of this food item?",
        "Q5: What movie do you think of when thinking of this food item?",
        "Q6: What drink would you pair with this food item?",
        "Q7: When you think about this food item, who does it remind you of?",
        "Q8: How much hot sauce would you add to this food item?"
    ]

    num_cols = [
        "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
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
        # "Q7: When you think about this food item, who does it remind you of?",
        # "Q8: How much hot sauce would you add to this food item?"
    ]
    text_columns = [
        # "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
        "Q2: How many ingredients would you expect this food item to contain?",
        # "Q3: In what setting would you expect this food to be served? Please check all that apply",
        # "Q4: How much would you expect to pay for one serving of this food item?",
        # "Q5: What movie do you think of when thinking of this food item?",
        # "Q6: What drink would you pair with this food item?",
        # "Q7: When you think about this food item, who does it remind you of?",
        # "Q8: How much hot sauce would you add to this food item?"
    ]
    prob_selection_rev = {
        "text":[
            # "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
            # "Q2: How many ingredients would you expect this food item to contain?",
            # "Q3: In what setting would you expect this food to be served? Please check all that apply",
            # "Q4: How much would you expect to pay for one serving of this food item?",
            # "Q5: What movie do you think of when thinking of this food item?",
            # "Q6: What drink would you pair with this food item?",
            # "Q7: When you think about this food item, who does it remind you of?",
            # "Q8: How much hot sauce would you add to this food item?"
        ]
        ,
        "cate":[
            # "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
            # "Q2: How many ingredients would you expect this food item to contain?",
            # "Q3: In what setting would you expect this food to be served? Please check all that apply",
            # "Q4: How much would you expect to pay for one serving of this food item?",
            # "Q5: What movie do you think of when thinking of this food item?",
            # "Q6: What drink would you pair with this food item?",
            # "Q7: When you think about this food item, who does it remind you of?",
            # "Q8: How much hot sauce would you add to this food item?"
        ]
    }
    def convert_prob_selection(prob_selection_rev, all_features):
        new_mapping = {}
        for f in all_features:
            if f in prob_selection_rev.get("text", []):
                new_mapping[f] = "text"
            elif f in prob_selection_rev.get("cate", []):
                new_mapping[f] = "cate"
            else:
                new_mapping[f] = "none"
        return new_mapping
    prob_selection = convert_prob_selection(prob_selection_rev, features)

    file_name = "cleaned_data_combined.csv"
    data_path = Path.cwd().parent / file_name
    df = pd.read_csv(data_path)
    df_shuffled = df.sample(frac=1, random_state=42)  # Shuffle rows (set random_state for reproducibility)
    df_train = df_shuffled.iloc[:int(len(df_shuffled) * 0.8)]  # Take first 80% of rows
    df_test = df_shuffled.iloc[int(len(df_shuffled) * 0.8):]


    best_results = feature_selection_best_combination(df_train,
                                       candidate_numeric_features = num_cols,
                                       candidate_text_features = text_columns,
                                       candidate_categorical_features = cate_cols,
                                       prob_dict= prob_selection,
                                       n_splits=5,
                                       output_config_file="best_config.pkl",
                                       loop_numeric=True,
                                       loop_text=True,
                                       loop_categorical=True)
    print("best results:", best_results)

    print("Testing combination:")
    print("  Numeric:", num_cols)
    print("  Text:", text_columns)
    print("  Categorical:", cate_cols)
    hmodel = HybridModel(
        num_cols=num_cols,
        text_cols=text_columns,
        prob_dict=prob_selection,
        cate_cols=cate_cols,
        label_col="Label",
    )
    # best_features = hmodel.optimize_feature_selection(df_train)
    print(cross_validation_test(df_test, hmodel, n_splits=5))



    #
    # with open("best_config.pkl", 'rb') as f:
    #     best_features = pickle.load(f)
    # hmodel = HybridModel(
    #     num_cols=best_features["final_num_cols"],
    #     text_cols=best_features["final_text_cols"],
    #     cate_cols=best_features["final_cate_cols"],
    #     prob_cols=best_features["final_prob_cols"],
    #     label_col="Label",
    #     last_model=LogisticRegression(max_iter=10000)
    # )
    # hmodel.train(df)
    # hmodel.save("model_params.pkl")





