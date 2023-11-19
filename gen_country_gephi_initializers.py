import csv
from itertools import combinations
from collections import defaultdict
import pandas as pd
import json
import os

NETFLIX_DATA = "data/netflix_titles.csv"
os.makedirs("generated", exist_ok=True)
DATE_ADDED_COLUMN_NAME = "date_added"

# Country
COUNTRY = "India"


def main():
    netflix_data = pd.read_csv(NETFLIX_DATA)
    ##### POPULATE MOVIES ACTORS (FILTER BY COUNTRY) #####
    movies_actors = {}
    for movie, actors, countries in zip(
        netflix_data["title"], netflix_data["cast"], netflix_data["country"]
    ):
        if not isinstance(countries, str):
            continue
        countries = [x.strip() for x in countries.split(",")]
        if isinstance(actors, str) and COUNTRY in countries:
            movies_actors[movie] = [x.strip() for x in actors.split(",")]
            continue
        movies_actors[movie] = []

    ##### POPULATE ALL ACTORS IN COUNTRY #####
    all_actors = set()
    for actors in movies_actors.values():
        all_actors = all_actors.union(set(actors))
    all_actors = list(all_actors)

    ##### POPULATE ACTOR DETAILS #####
    actor_details = {}
    with open("data/col_4_actors_data.json", "r") as file:
        actor_details = json.load(file)

    ##### POPULATE ACTOR TO POPULARITY #####
    actor_to_popularity = {}
    for actor, details in actor_details.items():
        popularity = details.get("popularity", 0)
        if (
            popularity == 0
        ):  # If popularity is 0, set it to 0.5 so at least it shows up in the graph
            popularity = 0.5
        actor_to_popularity[actor] = popularity
    actor_to_popularity = {
        k: v for k, v in actor_to_popularity.items() if k in all_actors
    }

    ##### GENERATE CSV FILES FOR GEPHI #####
    gen_gephi_edges(movies_actors, f"generated/{COUNTRY}-gephi-edges.csv")
    gen_gephi_id_label_popularity(
        actor_to_popularity, f"generated/{COUNTRY}-gephi-id_label_popularity.csv"
    )


def gen_gephi_edges(movies_actors, output_filename):
    """Generate CSV file for Gephi to read in edges."""
    edges_set = set()

    for cast in movies_actors.values():
        for pair in combinations(cast, 2):  # combinations for undirected graph
            sorted_pair = tuple(sorted(pair))  # sort to avoid duplicates
            edges_set.add(sorted_pair)

    edges_list = list(edges_set)

    with open(output_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Source", "Target"])  # Header row
        for edge in edges_list:
            writer.writerow(edge)


def gen_gephi_id_label_popularity(actor_to_popularity, output_filename):
    """Generate CSV file for Gephi to read in nodes."""
    with open(output_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "Label", "Popularity"])
        for actor, popularity in actor_to_popularity.items():
            writer.writerow([actor, actor, popularity])


if __name__ == "__main__":
    main()
