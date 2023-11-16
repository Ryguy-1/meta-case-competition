import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from itertools import combinations
import csv
from llm_utils import *
from utils import *
import os
import json
import time
import concurrent
import numpy as np

NETFLIX_DATA = "data/netflix_titles.csv"
os.makedirs("generated", exist_ok=True)


def main():
    netflix_data = pd.read_csv(NETFLIX_DATA)
    # gen_type_column_graphs(netflix_data)
    # gen_title_column_graphs(netflix_data)
    # gen_director_column_graphs(netflix_data)
    gen_cast_column_gephi_tables(netflix_data)


def gen_type_column_graphs(data):
    """Column: type"""
    type_counts = data["type"].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=type_counts.index, y=type_counts.values, palette="viridis")
    plt.title("Distribution of Netflix Titles: Movies vs TV Shows")
    plt.xlabel("Type")
    plt.ylabel("Count")
    plt.tight_layout()
    save_plot(plt, "generated/col_1_type.png")


def gen_title_column_graphs(data):
    """Column: title"""
    # WORDCLOUD
    titles_text = " ".join(data["title"].dropna())
    wordcloud_custom = WordCloud(
        width=800,
        height=400,
        background_color="black",
        colormap="plasma",  # Using a modern color scheme
        contour_color="steelblue",
        contour_width=3,
        max_words=200,
        random_state=42,
    ).generate(titles_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_custom, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Netflix Titles", color="white")
    plt.tight_layout()
    save_plot(plt, "generated/col_2_title_wordcloud.png")


def gen_director_column_graphs(data):
    """Column: director"""
    director_counts = data["director"].dropna().value_counts()
    top_directors = director_counts.head(20)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_directors.values, y=top_directors.index, palette="coolwarm")
    plt.title("Top 20 Directors on Netflix")
    plt.xlabel("Number of Titles")
    plt.ylabel("Director")
    plt.tight_layout()
    save_plot(plt, "generated/col_3_director_popular.png")


def gen_cast_column_gephi_tables(data):
    """
    Column: cast

    Note: We use Gephi for this column to generate a network graph.
    """
    movies_actors = {}
    for movie, actors in zip(data["title"], data["cast"]):
        if isinstance(actors, str):
            movies_actors[movie] = actors.split(", ")
            continue
        movies_actors[movie] = []

    # Generate Edge Set for Gephi
    edges_set = set()

    for cast in movies_actors.values():
        for pair in combinations(cast, 2):  # combinations for undirected graph
            sorted_pair = tuple(sorted(pair))  # sort to avoid duplicates
            edges_set.add(sorted_pair)

    edges_list = list(edges_set)

    csv_filename = "generated/col_4_gephi_cast_edges.csv"
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Source", "Target"])  # Header row
        for edge in edges_list:
            writer.writerow(edge)

    # Get Each Actor's Stats
    all_actors = list(set([actor for cast in movies_actors.values() for actor in cast]))
    futures = {}
    actor_details = {}

    # Load Actor Data from File (initialize)
    if os.path.exists("generated/col_4_actors_data.json"):
        with open("generated/col_4_actors_data.json", "r") as file:
            actor_details = json.load(file)

    def save_current_actor_data(these_stats):
        with open("generated/col_4_actors_data.json", "w") as file:
            json.dump(these_stats, file, indent=4)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for actor in all_actors:
            if actor in actor_details:
                print(f"Skipping {actor}. Already got their data.")
                continue
            future = executor.submit(get_actor_details, actor)
            futures[future] = actor
        while futures:
            done, _ = concurrent.futures.wait(
                futures, timeout=3, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                actor = futures.pop(future)
                actor_details[actor] = future.result()
            save_current_actor_data(actor_details)
            time.sleep(60)
            print(f"Got {len(actor_details)}/{len(all_actors)} actors' data so far...")
    save_current_actor_data(actor_details)

    # Pull Popularities and Create Nodes Table for Gephi
    node_to_popularity = {}
    for actor, details in actor_details.items():
        popularity = details.get("popularity", 0)
        if (
            popularity == 0
        ):  # If popularity is 0, set it to 0.1 so at least it shows up in the graph
            popularity = 0.1
        node_to_popularity[actor] = popularity

    csv_filename = "generated/col_4_gephi_node_to_popularity.csv"
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "Label", "Popularity"])
        for actor, popularity in node_to_popularity.items():
            writer.writerow([actor, actor, popularity])

    csv_filename = "generated/col_4_gephi_node_to_popularity_top_10_per_nationality.csv"
    actors_by_country_with_popularities = {}
    for actor, details in actor_details.items():
        country = details.get("place_of_birth", "")
        if country in ["", None]:
            continue
        country = country.split(",")[-1].strip()  # Ensure we only get the country
        if country not in actors_by_country_with_popularities:
            actors_by_country_with_popularities[country] = []
        actors_by_country_with_popularities[country].append(
            (actor, node_to_popularity[actor])
        )
    top_n_actors_by_country = {}
    for country, actors in actors_by_country_with_popularities.items():
        actors.sort(key=lambda x: x[1], reverse=True)
        top_n_actors_by_country[country] = actors[:10]

    # write to csv. if is top actor, put label, otherwise put empty string
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "Label", "Popularity"])
        for actor, popularity in node_to_popularity.items():
            country = actor_details[actor].get("place_of_birth", "")
            if country in ["", None]:
                continue
            country = country.split(",")[-1].strip()
            if (actor, popularity) in top_n_actors_by_country[country]:
                writer.writerow([actor, actor, popularity])
            else:
                writer.writerow([actor, "", popularity])


if __name__ == "__main__":
    main()
