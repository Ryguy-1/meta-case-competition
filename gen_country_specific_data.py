"""Generates data related to a specific country."""

import pandas as pd
import os
import json
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

NETFLIX_DATA = "data/netflix_titles.csv"
os.makedirs("generated", exist_ok=True)
DATE_ADDED_COLUMN_NAME = "date_added"
COUNTRY_NAME = "Spain"


def main():
    netflix_data = pd.read_csv(NETFLIX_DATA)
    # Generate plots
    country_films = netflix_data[
        netflix_data["country"].apply(lambda x: COUNTRY_NAME in str(x).split(","))
    ]
    with open("data/movie_data.json", "r") as file:
        genre_data = json.load(file)
    genre_data = {
        k: [g["name"] for g in v.get("genres", [])] for k, v in genre_data.items()
    }
    with open("data/col_4_actors_data.json", "r") as file:
        actors_data = json.load(file)

    generate_rating_distribution_plot(
        country_films, f"generated/rating_distribution_{COUNTRY_NAME.lower()}.png"
    )
    generate_genre_distribution_plot(
        df=country_films,
        genres_data=genre_data,
        save_path=f"generated/genre_distribution_{COUNTRY_NAME.lower()}.png",
    )
    generate_top_cast_members_plot(
        df=country_films,
        save_path=f"generated/top_cast_members_{COUNTRY_NAME.lower()}.png",
    )
    generate_actor_popularity_over_time(
        df=country_films,
        actor_data=actors_data,
        save_path=f"generated/actor_popularity_over_time_{COUNTRY_NAME.lower()}.png",
    )
    generate_actor_birthplace_distribution(
        df=country_films,
        actor_data=actors_data,
        save_path=f"generated/actor_birthplace_distribution_{COUNTRY_NAME.lower()}.png",
    )


def generate_rating_distribution_plot(df, save_path):
    """Generates and saves a sorted stacked area plot of rating distribution over time as a percentage of the total."""
    df = df.copy()
    df["year_added"] = pd.to_datetime(
        df[DATE_ADDED_COLUMN_NAME].str.strip(), errors="coerce"
    ).dt.year
    rating_over_time = df.groupby(["year_added", "rating"]).size().unstack(fill_value=0)

    # Normalizing data to get percentages
    rating_over_time_percentage = (
        rating_over_time.div(rating_over_time.sum(axis=1), axis=0) * 100
    )

    # Sorting keys by total percentage contribution
    sorted_keys = rating_over_time_percentage.sum().sort_values(ascending=False).index
    rating_over_time_percentage = rating_over_time_percentage[sorted_keys]

    plt.figure(figsize=(12, 6))
    plt.stackplot(
        rating_over_time_percentage.index,
        rating_over_time_percentage.T,
        colors=sns.color_palette("tab20", len(rating_over_time_percentage.columns)),
    )
    plt.title("Rating Distribution Over Time (Percentage, Sorted)")
    plt.xlabel("Year")
    plt.ylabel("Percentage of Films")
    plt.legend(loc="upper left", labels=sorted_keys)
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_genre_distribution_plot(df, genres_data, save_path):
    """Generates and saves a sorted stacked area plot of genre distribution over time as a percentage of the total."""
    df = df.copy()
    df["year_added"] = pd.to_datetime(
        df[DATE_ADDED_COLUMN_NAME].str.strip(), errors="coerce"
    ).dt.year
    df["genres"] = df["title"].map(genres_data).fillna("Unknown")
    df_exploded = df.explode("genres")
    genre_over_time = (
        df_exploded.groupby(["year_added", "genres"]).size().unstack(fill_value=0)
    )

    # Normalizing data to get percentages
    genre_over_time_percentage = (
        genre_over_time.div(genre_over_time.sum(axis=1), axis=0) * 100
    )

    # Sorting keys by total percentage contribution
    sorted_genres = (
        genre_over_time_percentage.sum().sort_values(ascending=False).head(20).index
    )
    genre_over_time_percentage = genre_over_time_percentage[sorted_genres]

    plt.figure(figsize=(12, 6))
    plt.stackplot(
        genre_over_time_percentage.index,
        genre_over_time_percentage.T,
        colors=sns.color_palette("tab20", len(genre_over_time_percentage.columns)),
    )
    plt.title("Genre Distribution Over Time (Percentage, Sorted)")
    plt.xlabel("Year")
    plt.ylabel("Percentage of Films")
    plt.legend(loc="upper left", fontsize="small", labels=sorted_genres)
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_top_cast_members_plot(df, save_path):
    """Generates and saves a bar plot of the top 10 most frequent cast members in Spanish films."""

    # Function to split the cast string into individual names
    def split_cast(cast_string):
        if pd.isna(cast_string):
            return []
        return [name.strip() for name in cast_string.split(",")]

    # Filtering for Spanish films and expanding the cast into separate rows
    country_films_cast = df["cast"].apply(split_cast).explode()

    # Counting the frequency of each cast member
    cast_frequency = country_films_cast.value_counts().head(10)

    # Plotting the top 10 most frequent cast members
    plt.figure(figsize=(12, 6))
    sns.barplot(x=cast_frequency.values, y=cast_frequency.index, palette="crest")
    plt.title("Top 10 Most Frequent Cast Members in Spanish Films")
    plt.xlabel("Number of Films")
    plt.ylabel("Cast Member")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_actor_popularity_over_time(df, actor_data, save_path):
    """Generates and saves a scatter plot of actor popularity over time."""
    # Filtering actor data to include only those who appeared in Spanish films
    spanish_actors = set(df["cast"].dropna().str.split(",").explode().str.strip())
    filtered_actors = {
        name: info for name, info in actor_data.items() if name in spanish_actors
    }

    # Extracting data
    popularity_data = [
        (
            actor["name"],
            actor.get("popularity", 0),
            pd.to_datetime(actor.get("birthday")),
        )
        for actor in filtered_actors.values()
        if actor.get("birthday")
    ]
    df_popularity = pd.DataFrame(
        popularity_data, columns=["Name", "Popularity", "Birthday"]
    )
    df_popularity["Year"] = df_popularity["Birthday"].dt.year

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=df_popularity, x="Year", y="Popularity", hue="Name", legend=False
    )
    plt.title("Actor Popularity Over Time in Spanish Films")
    plt.xlabel("Year of Birth")
    plt.ylabel("Popularity")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_actor_birthplace_distribution(df, actor_data, save_path):
    """Generates and saves a bar chart of actor birthplace distribution."""
    # Filtering actor data to include only those who appeared in Spanish films
    spanish_actors = set(df["cast"].dropna().str.split(",").explode().str.strip())
    filtered_actors = {
        name: info for name, info in actor_data.items() if name in spanish_actors
    }

    # Extracting data
    birthplace_data = [
        actor.get("place_of_birth", "Unknown").split(", ")[-1]
        for actor in filtered_actors.values()
        if actor.get("place_of_birth")
    ]
    df_birthplace = pd.DataFrame(birthplace_data, columns=["Country"])
    country_counts = df_birthplace["Country"].value_counts()

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.barplot(x=country_counts.values, y=country_counts.index, palette="viridis")
    plt.title("Birthplace Country Distribution of Actors in Spanish Films")
    plt.xlabel("Number of Actors")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    main()
