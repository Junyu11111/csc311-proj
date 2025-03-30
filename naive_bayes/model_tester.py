
from sklearn.naive_bayes import MultinomialNB

from naive_bayes.feature_selection import *


def evaluate_feature_models(X, y, models_list,
                            cv=5, test_size=0.2, random_state=42, n_iter=5):



    # Split the data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    results = {}

    # Loop through each model configuration.
    for model_info in models_list:
        model_name = model_info.get("name", "Unnamed Model")
        estimator = model_info["estimator"]
        search_space = model_info["search_space"]


        print(f"\nTuning model: {model_name}")
        # CV to find the best hyperparams
        optimizer = BayesSearchCV(
            estimator=estimator,
            search_spaces=search_space,
            n_iter=n_iter,
            cv=cv,
            scoring='accuracy',
            random_state=random_state,
            n_jobs=1
        )
        optimizer.fit(X_train, y_train)

        best_params = optimizer.best_params_
        best_cv_score = optimizer.best_score_
        test_score = optimizer.score(X_test, y_test)

        model_result = {
            "model_name": model_name,
            "best_params": best_params,
            "best_cv_score": best_cv_score,
            "test_score": test_score
        }
        results[model_name] = model_result

        print(f"{model_name} - Best CV Score: {best_cv_score:.4f}, "
              f"Test Score: {test_score:.4f}, Best Params: {best_params}")

    return results


def evaluate_features(feature_list, feature_fn, feature_type, df, y, models):
    """
    Evaluates a list of features using a given feature extraction function,
    and aggregates the results with the feature name and feature type.
    """
    aggregated_results = []
    for feature in feature_list:
        X = feature_fn(df, feature)
        model_results = evaluate_feature_models(X, y, models)
        for model_name, res in model_results.items():
            res_update = res.copy()
            res_update.update({"feature": feature, "feature_type": feature_type})
            aggregated_results.append(res_update)
    return aggregated_results


# Example usage:
if __name__ == "__main__":
    # Load your dataset.
    file_name = "cleaned_data_combined.csv"
    data_path = Path.cwd().parent / file_name
    df = pd.read_csv(data_path)

    # Specify the label column.
    label_column = "Label"

    # Define the list of models with their estimators and search spaces.
    models_to_test = [
        {
            "name": "Naive Bayes",
            "estimator": MultinomialNB(),  # from sklearn.naive_bayes
            "search_space": {
                "alpha": (1e-3, 1e+2, "log-uniform")
            }
        },
        {
            "name": "Logistic Regression",
            "estimator": LogisticRegression(max_iter=10000, solver='liblinear'),  # from sklearn.linear_model
            "search_space": {
                "C": (1e-3, 1e4, "log-uniform")
            }
        },
        {
            "name": "Random Forest",
            "estimator": RandomForestClassifier(),  # from sklearn.ensemble
            "search_space": {
                "n_estimators": (50, 600, "uniform"),
                "max_depth": (2, 100, "uniform"),
                "min_samples_split": (2, 10, "uniform")
            }
        },
        {
            "name": "KNN",
            "estimator": KNeighborsClassifier(),  # from sklearn.neighbors
            "search_space": {
                "n_neighbors": (1, 30, "uniform"),
                "weights": ["uniform", "distance"],
                "p": (1, 2)  # 1 = Manhattan, 2 = Euclidean
            }
        }
    ]

    candidate_num_cols = [
        "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
        "Q2: How many ingredients would you expect this food item to contain?",
        "Q3: In what setting would you expect this food to be served? Please check all that apply",
        "Q4: How much would you expect to pay for one serving of this food item?",
        "Q5: What movie do you think of when thinking of this food item?",
        "Q6: What drink would you pair with this food item?",
        "Q7: When you think about this food item, who does it remind you of?",
        "Q8: How much hot sauce would you add to this food item?"
    ]
    candidate_text_cols = [
        "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
        "Q2: How many ingredients would you expect this food item to contain?",
        "Q3: In what setting would you expect this food to be served? Please check all that apply",
        "Q4: How much would you expect to pay for one serving of this food item?",
        "Q5: What movie do you think of when thinking of this food item?",
        "Q6: What drink would you pair with this food item?",
        "Q7: When you think about this food item, who does it remind you of?",
        "Q8: How much hot sauce would you add to this food item?"
    ]
    candidate_category_cols = [
        "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
        "Q2: How many ingredients would you expect this food item to contain?",
        "Q3: In what setting would you expect this food to be served? Please check all that apply",
        "Q4: How much would you expect to pay for one serving of this food item?",
        "Q5: What movie do you think of when thinking of this food item?",
        "Q6: What drink would you pair with this food item?",
        "Q7: When you think about this food item, who does it remind you of?",
        "Q8: How much hot sauce would you add to this food item?"
    ]
    y = create_t(df, label_column)
    all_results = []
    all_results += evaluate_features(candidate_num_cols, create_num_features, "numerical", df, y, models_to_test)
    all_results += evaluate_features(candidate_category_cols, create_category_features, "category", df, y, models_to_test)
    all_results += evaluate_features(candidate_text_cols, create_text_features, "text", df, y, models_to_test)

    # Convert the results to a DataFrame and save to CSV.
    results_df = pd.DataFrame(all_results)
    results_csv_path = Path("evaluation_results.csv")
    results_df.to_csv(results_csv_path, index=False)



