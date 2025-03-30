import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy.utilities.autowrap import autowrap


def plot_evaluation_results(csv_file_path, output_image_path):

    """
    Reads the evaluation CSV and creates one subplot per feature.
    In each subplot, the x-axis is the model name and the y-axis is the test accuracy.
    If the same feature has multiple representations (feature_type), their test scores are shown as grouped bars.
    The resulting figure is saved to output_image_path.
    """
    df = pd.read_csv(csv_file_path)

    cols = {
        "id": "id",
        "complexity": "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
        "ingredients": "Q2: How many ingredients would you expect this food item to contain?",
        "setting": "Q3: In what setting would you expect this food to be served? Please check all that apply",
        "price": "Q4: How much would you expect to pay for one serving of this food item?",
        "movie": "Q5: What movie do you think of when thinking of this food item?",
        "drink": "Q6: What drink would you pair with this food item?",
        "reminder": "Q7: When you think about this food item, who does it remind you of?",
        "hotsauce": "Q8: How much hot sauce would you add to this food item?",
        "label": "Label"
    }
    rev_col= {v: k for k, v in cols.items()}

    color_map = {
        "category": "blue",
        "text": "orange",
        "numerical": "green"
    }

    # Get unique features (each subplot will represent one feature).
    unique_features = df["feature"].unique()
    num_features = len(unique_features)

    # Set up a grid for subplots.
    ncols = 3
    nrows = math.ceil(num_features / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)

    for idx, feature in enumerate(unique_features):
        ax = axes[idx // ncols][idx % ncols]
        subset = df[df["feature"] == feature]

        # Unique models evaluated on this feature.
        models = subset["model_name"].unique()
        # Unique representations (feature_type) for this feature.
        feature_types = subset["feature_type"].unique()

        # Define bar width so that bars for multiple representations fit under one model.
        bar_width = 0.8 / len(feature_types)
        indices = np.arange(len(models))

        # For each representation type, plot its corresponding test score for each model.
        for j, ft in enumerate(feature_types):
            scores = []
            for model in models:
                # Get row(s) for the given model and representation.
                row = subset[(subset["model_name"] == model) & (subset["feature_type"] == ft)]
                if not row.empty:
                    # If multiple entries exist, take the average test score.
                    scores.append(row["test_score"].mean())
                else:
                    scores.append(np.nan)
            ax.bar(indices + j * bar_width, scores, bar_width, label=ft, color=color_map[ft])

        ax.set_title(rev_col[feature])
        ax.set_xticks(indices + bar_width * (len(feature_types) - 1) / 2)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel("Test Accuracy")
        ax.set_ylim(0, 1)
        ax.legend(title="Representation")

    # Remove empty subplots if any.
    total_plots = nrows * ncols
    if num_features < total_plots:
        for idx in range(num_features, total_plots):
            fig.delaxes(axes[idx // ncols][idx % ncols])

    plt.tight_layout()
    plt.savefig(output_image_path)
    print(f"Plot saved as {Path(output_image_path).resolve()}")
    plt.show()


if __name__ == "__main__":
    # Path to the CSV file generated by evaluation code.
    csv_file_path = "evaluation_results.csv"
    output_image_path = "evaluation_plots.png"
    plot_evaluation_results(csv_file_path, output_image_path)
