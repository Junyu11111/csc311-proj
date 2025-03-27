from scipy.optimize import minimize

from feature_selection import *

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

if __name__ == "__main__":
    file_name = "cleaned_data_combined.csv"
    data_path = Path.cwd().parent / file_name
    random_state = 42
    train_percent = 0.8
    # Read the CSV file from the parent directory.

    print("Data file:", data_path)
    df = pd.read_csv(data_path)


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
    label_column = "Label"
    hyperparams_ab = {}
    for feature in feature_columns:
        print("Feature:", feature)
        X = create_text_features(df, feature, train_percent)
        t = create_t(df, label_column)
        hyperparams_ab[feature] = t

        N = X.shape[0]
        indices = np.arange(N)
        np.random.seed(42)
        np.random.shuffle(indices)

        train_size = int(train_percent * N)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        X_train, X_test = X[train_indices], X[test_indices]
        t_train, t_test = t[train_indices], t[test_indices]

        labels = df[label_column].fillna("").astype(str)
        unique_labels = sorted({label.strip() for label in labels})
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

        n_labels = len(label_mapping)
        def objective(params):
            x, y = params
            a = np.exp(x)  # a is always > 0
            b = np.exp(y)  # b is always > 0
            pi, theta = naive_bayes_map(X_train, t_train, n_labels, a_feat=a, b_feat=b)
            test_preds = make_prediction(X_test, pi, theta)
            test_acc = accuracy(test_preds, t_test)
            return -test_acc
        initial_guess = np.array([0.1, 0.1])
        result = minimize(objective, x0=initial_guess, method='Powell')
        best_params = np.exp(result.x)
        hyperparams_ab[feature]=best_params.tolist()

        pi, theta = naive_bayes_map(X_train, t_train, n_labels, a_feat=best_params[0], b_feat=best_params[1])
        test_preds = make_prediction(X_test, pi, theta)
        test_acc = accuracy(test_preds, t_test)
        print("Naive Bayes Valid Acc:", test_acc)
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, t_train)
        val_acc = lr.score(X_test, t_test)
        print("LR Valid Acc:", val_acc)
    print(hyperparams_ab)
