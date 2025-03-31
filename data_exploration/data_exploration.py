import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

def plot_numeric_features_correlation(df):
    df_numeric = df.copy()
    df_numeric.rename(columns={
        "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)": "Complexity"
    }, inplace=True)
    df_numeric["Ingredients"] = df["Q2: How many ingredients would you expect this food item to contain?"].str.extract("(\\d+)")
    df_numeric["Ingredients"] = pd.to_numeric(df_numeric["Ingredients"], errors="coerce")
    df_numeric["Price"] = df["Q4: How much would you expect to pay for one serving of this food item?"].str.extract("(\\d+)")
    df_numeric["Price"] = pd.to_numeric(df_numeric["Price"], errors="coerce")

    # Compute correlation matrix for numerical features
    correlation_matrix = df_numeric[["Complexity", "Ingredients", "Price"]].corr()

    # Plot the correlation heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Numerical Features")
    plt.show()


if __name__ == "__main__":
    file_name = "cleaned_data_combined_modified.csv"
    df = pd.read_csv(file_name)
    plot_ingredients_distribution(df)
    plot_price_distribution(df)
    plot_numeric_features_correlation(df)
