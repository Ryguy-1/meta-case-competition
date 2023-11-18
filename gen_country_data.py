"""Generates all data related to titles."""

import pandas as pd
from llm_utils import *
from utils import *
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl


NETFLIX_DATA = "data/netflix_titles.csv"
os.makedirs("generated", exist_ok=True)
DATE_ADDED_COLUMN_NAME = "date_added"


def main():
    netflix_data = pd.read_csv(NETFLIX_DATA)

    ##### Generate Title Counts by Country #####
    gen_num_per_country_over_time(
        data=netflix_data, output_filename="generated/title_counts_by_country.png"
    )


def gen_num_per_country_over_time(data, output_filename):
    """Generates Map of Number of Titles per Country Over Time."""
    # Preprocessing the data
    # Filling missing values in 'country' and splitting multi-country entries
    data["country"] = data["country"].dropna().apply(lambda x: x.strip())
    netflix_data_expanded = data.drop("country", axis=1).join(
        data["country"]
        .str.split(",", expand=True)
        .stack()
        .reset_index(level=1, drop=True)
        .rename("country")
    )
    netflix_data_expanded["country"] = (
        netflix_data_expanded["country"].str.strip().str.title()
    )

    # Extracting and cleaning year data
    netflix_data_expanded["year_added"] = pd.to_datetime(
        netflix_data_expanded["date_added"], errors="coerce"
    ).dt.year
    netflix_data_expanded = netflix_data_expanded.dropna(subset=["year_added"])
    netflix_data_expanded["year_added"] = netflix_data_expanded["year_added"].astype(
        int
    )

    # Counting titles per country per year
    titles_per_country_year = (
        netflix_data_expanded.groupby(["country", "year_added"])
        .size()
        .reset_index(name="number_of_titles")
    )

    # Sorting countries by total number of titles
    total_titles_per_country = titles_per_country_year.groupby("country")[
        "number_of_titles"
    ].sum()
    sorted_countries = total_titles_per_country.sort_values(
        ascending=False
    ).index.tolist()

    # Setting up the plot with a dark theme
    sns.set(style="darkgrid")
    plt.figure(figsize=(15, 8), dpi=120)

    # Generating random colors
    num_countries = len(sorted_countries)
    colors = np.random.choice(
        list(mpl.colors.XKCD_COLORS.keys()), num_countries, replace=False
    )

    # Plotting data with custom order and colors
    for country, color in zip(sorted_countries, colors):
        country_data = titles_per_country_year[
            titles_per_country_year["country"] == country
        ]
        plt.plot(
            country_data["year_added"],
            country_data["number_of_titles"],
            marker="o",
            label=country,
            color=color,
        )

    # Customizing the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    num_entries = 20  # Adjust this number as needed
    plt.legend(
        handles[:num_entries],
        labels[:num_entries],
        loc="upper left",
        fontsize=10,
        title="Country",
    )

    # Finalizing the plot
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Number of Titles", fontsize=12)
    plt.title("Number of Netflix Titles per Country Over Time", fontsize=14)
    plt.tight_layout()
    plt.savefig("generated/title_counts_by_country.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
