import pandas as pd
from itertools import combinations
import csv
from llm_utils import *
from utils import *
import os
import json


NETFLIX_DATA = "data/netflix_titles.csv"
os.makedirs("generated", exist_ok=True)
DATE_ADDED_COLUMN_NAME = "date_added"


def main():
    netflix_data = pd.read_csv(NETFLIX_DATA)

    ##### POPULATE MOVIES ACTORS #####
    movies_actors = {}
    for movie, actors in zip(netflix_data["title"], netflix_data["cast"]):
        if isinstance(actors, str):
            movies_actors[movie] = [x.strip() for x in actors.split(",")]
            continue
        movies_actors[movie] = []

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

    ##### POPULATE ACTOR TO MODULARITY CLASS #####
    actor_to_modularity_class = pd.read_csv("data/gephi_10_modularities.csv")
    actor_to_modularity_class = {
        row["Id"]: row["modularity_class"]
        for _, row in actor_to_modularity_class.iterrows()
    }  # Map actor to modularity class

    ##### GENERATE GEPHI CSV FILES #####
    gen_gephi_edges(
        movies_actors=movies_actors,
        output_filename="generated/col_4_gephi_cast_edges.csv",
    )
    gen_gephi_id_label_popularity(
        actor_to_popularity=actor_to_popularity,
        output_filename="generated/col_4_gephi_id_label_popularity.csv",
    )
    gen_gephi_id_label_popularity_label_conditional_on_modularity_class(
        actor_to_popularity=actor_to_popularity,
        actor_to_modularity_class=actor_to_modularity_class,
        output_filename="generated/col_4_gephi_id_label_popularity_modular_labels.csv",
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
    with open(output_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "Label", "Popularity"])
        for actor, popularity in actor_to_popularity.items():
            writer.writerow([actor, actor, popularity])


def gen_gephi_id_label_popularity_label_conditional_on_modularity_class(
    actor_to_popularity, actor_to_modularity_class, output_filename
):
    # Create Map from Modularity Class to List of (Actor, Popularity) Tuples
    modularity_class_to_actor_popularity_list = {}
    for actor, modularity_class in actor_to_modularity_class.items():
        if modularity_class not in modularity_class_to_actor_popularity_list:
            modularity_class_to_actor_popularity_list[modularity_class] = []
        modularity_class_to_actor_popularity_list[modularity_class].append(
            (actor, actor_to_popularity[actor])
        )

    # Filter Modularity Classes By Those with 1% or More of the Total Actors
    total_actors = len(actor_to_modularity_class)
    modularity_class_to_actor_popularity_list = {
        modularity_class: actor_popularity_list
        for modularity_class, actor_popularity_list in modularity_class_to_actor_popularity_list.items()
        if len(actor_popularity_list) >= total_actors * 0.01
    }

    # Sort each list by popularity and take top 10
    for (
        modularity_class,
        actor_popularity_list,
    ) in modularity_class_to_actor_popularity_list.items():
        modularity_class_to_actor_popularity_list[modularity_class] = sorted(
            actor_popularity_list, key=lambda x: x[1], reverse=True
        )[:10]

    # Get Just List of Actors in Top 10 Per Modularity Class
    top_actors = set()
    for actor_popularity_list in modularity_class_to_actor_popularity_list.values():
        for actor, _ in actor_popularity_list:
            top_actors.add(actor)

    # Create CSV
    with open(output_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "Label", "Popularity"])
        for actor, popularity in actor_to_popularity.items():
            if actor in top_actors:
                writer.writerow([actor, actor, popularity])
            else:
                writer.writerow([actor, "", popularity])


if __name__ == "__main__":
    main()
