import itertools

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV

from naive_bayes.Model import *


from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


class PercentageSVD:
    def __init__(self, percentage=0.05):
        self.percentage = percentage # percentage var
        self.svd = None

    def fit(self, X, y=None):
        # construct a temp svd to calculate the variance
        temp_svd = TruncatedSVD(n_components=min(500, X.shape[1]))
        temp_svd.fit(X)
        cumulative = np.cumsum(temp_svd.explained_variance_ratio_)
        # K such that first K has variance > percentage * total var
        k = np.searchsorted(cumulative, self.percentage) + 1
        self.svd = TruncatedSVD(n_components=k)
        self.svd.fit(X)

    def transform(self, X):
        return self.svd.transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {
            "percentage": self.percentage,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self




def train(df, xgen, tune_hyperparams=True, best_params=None):
    best_score = 0
    X_train = df.drop(columns=["Label"])
    print("X_train.shape:", X_train.shape)
    t_train = df["Label"]

    pipeline = Pipeline([
        ('transform', xgen),
        ('svd', PercentageSVD()),
        ('classifier', LogisticRegression(max_iter=10000))
    ])

    LG_search_space = {
        "C": (1e-3, 1e4, "log-uniform"),
    }

    search_space = {
        'transform__text_method': ['binary', 'tfidf'],
        'svd__percentage': (0.7, 1, "uniform"),
        **{f"classifier__{key}": value for key, value in LG_search_space.items()},
    }

    if tune_hyperparams:
        grid_search = BayesSearchCV(
            estimator=pipeline,
            search_spaces = search_space,
            # param_grid=param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=6
        )
        grid_search.fit(X_train, t_train)
        best_pipeline = grid_search.best_estimator_

        best_params = grid_search.best_params_
        print("Best params:", best_params)
        print("Best CV accuracy:", grid_search.best_score_)
        best_score = grid_search.best_score_
    else:
        if best_params is None:
            raise ValueError("When tune_hyperparams is False, best_params must be provided.")
        best_pipeline = pipeline.set_params(**best_params)
        best_pipeline.fit(X_train, t_train)
    model_params = {}
    svd = best_pipeline.named_steps['svd'].svd
    classifier = best_pipeline.named_steps['classifier']
    svd_components = svd.components_
    W = classifier.coef_
    b = classifier.intercept_
    classes = classifier.classes_
    # Store for reuse
    model_params = {
        'svd_components': svd_components,
        'W': W,
        'b': b,
        'classes': classes,
    }
    return model_params, best_pipeline, best_params, best_score

def feature_test(df, features_to_test, possible_assignment, num_cols, cate_cols, text_cols):
    features = features_to_test
    assignments = possible_assignment
    high_score = 0
    best_combo = None

    for combo in itertools.product(assignments, repeat=len(features)):
        print("testing combo:", combo)
        # Map feature assignments
        num_cols = [f for f, a in zip(features, combo) if a == 'num' or "both"] + num_cols
        text_cols = [f for f, a in zip(features, combo) if a == 'text' or "both"] + text_cols

        # Skip if nothing is selected
        if not num_cols and not text_cols:
            continue
        xgen = XGen(num_cols, cate_cols, text_cols, text_method="binary")
        _,_,_,best_score = train(df, xgen)

        if best_score > high_score:
            high_score = best_score
            best_combo = combo
            print("Best score:", best_score)
            print("Best combo:", best_combo)

    return best_combo






if __name__ == "__main__":
    df = pd.read_csv("naive_bayes/cleaned_data_combined.csv")
    df_shuffled = df.sample(frac=1, random_state=42)  # Shuffle rows (set random_state for reproducibility)
    df_train = df_shuffled.iloc[:int(len(df_shuffled) * 0.8)]  # Take first 80% of rows
    df_test = df_shuffled.iloc[int(len(df_shuffled) * 0.8):]
    print("df_train shape: ", df_train.shape)
    X_test = df_test
    t_test = df_test["Label"]
    X_train = df_train
    t_train = df_train["Label"]

    num_cols = [
        # "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
        # "Q2: How many ingredients would you expect this food item to contain?",
        #  "Q3: In what setting would you expect this food to be served? Please check all that apply",
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
    text_cols = [
        # "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
        "Q2: How many ingredients would you expect this food item to contain?",
        # "Q3: In what setting would you expect this food to be served? Please check all that apply",
        "Q4: How much would you expect to pay for one serving of this food item?",
        "Q5: What movie do you think of when thinking of this food item?",
        "Q6: What drink would you pair with this food item?",
        # "Q7: When you think about this food item, who does it remind you of?",
        # "Q8: How much hot sauce would you add to this food item?"
    ]
    features_to_test = [
        "Q2: How many ingredients would you expect this food item to contain?",
        "Q4: How much would you expect to pay for one serving of this food item?",
    ]


    xgen_filename = "naive_bayes/xgen_params.pkl"
    classifier_filename = "naive_bayes/classifier_params.pkl"
    xgen = XGen(num_cols, cate_cols, text_cols, text_method="binary")

    # Train on
    classifier_params, best_model, hyperparams, _= train(df_train, xgen)
    xgen = best_model.named_steps['transform']

    # save the resulting params
    # xgen.save(xgen_filename)
    # with open(classifier_filename, "wb") as f:
    #     pickle.dump(classifier_params, f)


    # best model given by CV_search
    print("train score", best_model.score(df_train, t_train))

    print("best model test score", best_model.score(df_test, t_test))

    # test if our own model works the same as the sklearn pipline
    y = predict_df(X_test, xgen_filename, classifier_filename)
    print("ModelClassifier test score", np.mean(y == t_test))

    # # Train the full model without changing hyperparams or evaluation
    # model_params, best_pipeline, _ = train(df_shuffled, xgen, tune_hyperparams=False, best_params=hyperparams)
    # xgen.save("col_params.pkl")
    # with open("model_params.pkl", "wb") as f:
    #     pickle.dump(model_params, f)





