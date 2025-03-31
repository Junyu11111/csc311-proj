import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.stats as stats

def plot_complexity_distribution(df):
    # Convert Q1 to numeric for analysis
    df_numeric = df.copy()
    df_numeric.rename(columns={
        "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)": "Complexity"
    }, inplace=True)

    # Plot distribution of Complexity ratings
    plt.figure(figsize=(8, 5))
    sns.countplot(x=df_numeric["Complexity"], palette="viridis")
    plt.title("Distribution of Complexity Ratings")
    plt.xlabel("Complexity (1 = Simple, 5 = Complex)")
    plt.ylabel("Count")
    plt.show()

def plot_ingredients_distribution(df):
    df_numeric = df.copy()
    df_numeric["Ingredients"] = df["Q2: How many ingredients would you expect this food item to contain?"].str.extract("(\\d+)")
    df_numeric["Ingredients"] = pd.to_numeric(df_numeric["Ingredients"], errors="coerce")

    plt.figure(figsize=(8, 5))
    sns.histplot(df_numeric["Ingredients"].dropna(), bins=15, kde=True, color="blue")
    plt.title("Distribution of Expected Ingredient Count")
    plt.xlabel("Number of Ingredients")
    plt.ylabel("Count")
    plt.show()

def plot_price_distribution(df):
    df_numeric = df.copy()
    df_numeric["Price"] = df["Q4: How much would you expect to pay for one serving of this food item?"].str.extract("(\\d+)")
    df_numeric["Price"] = pd.to_numeric(df_numeric["Price"], errors="coerce")

    plt.figure(figsize=(8, 5))
    sns.histplot(df_numeric["Price"].dropna(), bins=15, kde=True, color="green")
    plt.title("Distribution of Expected Price per Serving")
    plt.xlabel("Price ($)")
    plt.ylabel("Count")
    plt.show()


def plot_numeric_features_vs_label(df):
    df_numeric = df.copy()
    df_numeric.rename(columns={
        "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)": "Complexity",
        "Q2: How many ingredients would you expect this food item to contain?": "Ingredients",
        "Q4: How much would you expect to pay for one serving of this food item?": "Price"
    }, inplace=True)

    df_numeric["Ingredients"] = df_numeric["Ingredients"].str.extract("(\\d+)")
    df_numeric["Ingredients"] = pd.to_numeric(df_numeric["Ingredients"], errors="coerce")
    df_numeric["Price"] = df_numeric["Price"].str.extract("(\\d+)")
    df_numeric["Price"] = pd.to_numeric(df_numeric["Price"], errors="coerce")

    numerical_features = ["Complexity", "Ingredients", "Price"]
    plt.figure(figsize=(12, 5))

    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(1, 3, i)
        sns.boxplot(x=df_numeric["Label"], y=df_numeric[feature])
        plt.xticks(rotation=90)
        plt.title(f"{feature} by Label")

    plt.tight_layout()
    plt.show()

    # ANOVA test to check correlation between numerical features and Label
    for feature in numerical_features:
        groups = [df_numeric[df_numeric["Label"] == label][feature].dropna() for label in df_numeric["Label"].unique()]
        stat, p = stats.f_oneway(*groups)
        print(f"ANOVA for {feature}: F-statistic = {stat:.2f}, p-value = {p:.5f}")


if __name__ == "__main__":
    file_name = "cleaned_data_combined_modified.csv"
    df = pd.read_csv(file_name)
    plot_ingredients_distribution(df)
    plot_price_distribution(df)
    plot_numeric_features_vs_label(df)
