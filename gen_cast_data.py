"""Generates all data related to cast members."""

import pandas as pd
from itertools import combinations
import csv
from llm_utils import *
import os
import json
import datetime
import seaborn as sns
import matplotlib.pyplot as plt


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

    ##### POPULATE ACTOR TO AGE ##### (age stored like "birthday": "1975-02-09",)
    actor_to_age = {}
    for actor, details in actor_details.items():
        birthday = details.get("birthday", 0)
        if birthday == 0 or birthday is None:
            continue
        birthday = datetime.datetime.strptime(birthday, "%Y-%m-%d")
        today = datetime.datetime.today()
        age = (
            today.year
            - birthday.year
            - ((today.month, today.day) < (birthday.month, birthday.day))
        )
        actor_to_age[actor] = age

    ##### POPULATE ACTOR TO GENDER ##### (0=N/A, 1=F, 2=M)
    actor_to_gender = {}
    for actor, details in actor_details.items():
        gender = details.get("gender", 0)
        if gender == 1:
            actor_to_gender[actor] = "F"
        elif gender == 2:
            actor_to_gender[actor] = "M"

    ##### POPULATE ACTOR TO MODULARITY CLASS #####
    actor_to_modularity_class = pd.read_csv("data/gephi_10_modularities.csv")
    actor_to_modularity_class = {
        row["Id"]: row["modularity_class"]
        for _, row in actor_to_modularity_class.iterrows()
    }  # Map actor to modularity class

    ##### GENERATE CAST DATA GRAPHS OVER TIME #####
    gen_popularity_of_actors_over_time(
        netflix_data=netflix_data,
        actor_to_popularity=actor_to_popularity,
        output_filename="generated/col_4_popularity_of_actor_over_time.png",
    )
    gen_actor_age_over_time(
        netflix_data=netflix_data,
        actor_to_age=actor_to_age,
        output_filename="generated/col_4_age_of_actor_over_time.png",
    )

    gen_actor_gender_over_time(
        netflix_data=netflix_data,
        actor_to_gender=actor_to_gender,
        output_filename="generated/col_4_gender_of_actor_over_time.png",
    )
    generate_top_cast_graph(movies_actors, "generated/top_10_cast_members.png")

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


def gen_popularity_of_actors_over_time(
    netflix_data, actor_to_popularity, output_filename
):
    """Generate Graphs of Popularity of Actors Used Over Time."""
    # For each row in netflix data, get the date added (datetime -> year) and (actor -> popularity)
    # Note: format of date is currently like "September 25, 2021"
    year_to_actor_popularity = {}
    for _, row in netflix_data.iterrows():
        if not isinstance(row["cast"], str):
            continue
        if not isinstance(row[DATE_ADDED_COLUMN_NAME], str):
            continue
        date_added = datetime.datetime.strptime(
            row[DATE_ADDED_COLUMN_NAME].strip(), "%B %d, %Y"
        )
        actors_this_movie = [x.strip() for x in row["cast"].split(",")]
        if date_added.year not in year_to_actor_popularity:
            year_to_actor_popularity[date_added.year] = []
        year_to_actor_popularity[date_added.year] += [
            actor_to_popularity[actor] for actor in actors_this_movie
        ]
    year_to_avg_actor_popularity = {
        year: sum(popularities) / len(popularities)
        for year, popularities in year_to_actor_popularity.items()
    }
    # Plot
    sns.set_theme(style="darkgrid")
    sns.lineplot(
        x=list(year_to_avg_actor_popularity.keys()),
        y=list(year_to_avg_actor_popularity.values()),
    )
    plt.ylim(bottom=0)
    plt.xlabel("Year")
    plt.ylabel("Average Actor IMDB Popularity")
    plt.title("Average Actor IMDB Popularity Over Time")
    plt.savefig(output_filename)
    plt.close()


def gen_actor_age_over_time(netflix_data, actor_to_age, output_filename):
    """Generate Graphs of Age of Actors Used Over Time."""
    # For each row in netflix data, get the date added (datetime -> year) and (actor -> age)
    # Note: format of date is currently like "September 25, 2021"
    year_to_actor_age = {}
    for _, row in netflix_data.iterrows():
        if not isinstance(row["cast"], str):
            continue
        if not isinstance(row[DATE_ADDED_COLUMN_NAME], str):
            continue
        date_added = datetime.datetime.strptime(
            row[DATE_ADDED_COLUMN_NAME].strip(), "%B %d, %Y"
        )
        actors_this_movie = [x.strip() for x in row["cast"].split(",")]
        if date_added.year not in year_to_actor_age:
            year_to_actor_age[date_added.year] = []
        year_to_actor_age[date_added.year] += [
            actor_to_age[actor]
            for actor in actors_this_movie
            if actor in actor_to_age  # not all actors have age
        ]
    year_to_avg_actor_age = {
        year: sum(ages) / len(ages)
        for year, ages in year_to_actor_age.items()
        if len(ages) > 0  # not all years have actors with age
    }
    # Plot
    sns.set_theme(style="darkgrid")
    sns.lineplot(
        x=list(year_to_avg_actor_age.keys()), y=list(year_to_avg_actor_age.values())
    )
    plt.ylim(bottom=0)
    plt.xlabel("Year")
    plt.ylabel("Average Actor Age")
    plt.title("Average Actor Age Over Time")
    plt.savefig(output_filename)


def gen_actor_gender_over_time(netflix_data, actor_to_gender, output_filename):
    """Generate Graphs of Gender Distribution of Actors Used Over Time."""
    year_to_gender_count = {}
    for _, row in netflix_data.iterrows():
        if not isinstance(row["cast"], str):
            continue
        if not isinstance(row[DATE_ADDED_COLUMN_NAME], str):
            continue
        date_added = datetime.datetime.strptime(
            row[DATE_ADDED_COLUMN_NAME].strip(), "%B %d, %Y"
        )
        actors_this_movie = [x.strip() for x in row["cast"].split(",")]
        if date_added.year not in year_to_gender_count:
            year_to_gender_count[date_added.year] = {"F": 0, "M": 0}
        for actor in actors_this_movie:
            if actor in actor_to_gender:  # Check if actor's gender is known
                actor_gender = actor_to_gender[actor]
                year_to_gender_count[date_added.year][actor_gender] += 1

    # Convert the data into a format suitable for plotting
    years = []
    male_counts = []
    female_counts = []
    for year, genders in year_to_gender_count.items():
        years.append(year)
        male_counts.append(genders["M"])
        female_counts.append(genders["F"])

    # Plotting
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))
    plt.plot(years, male_counts, label="Male")
    plt.plot(years, female_counts, label="Female")
    plt.xlabel("Year")
    plt.ylabel("Count of Actors")
    plt.title("Gender Distribution of Actors Over Time")
    plt.legend()
    plt.savefig(output_filename)
    plt.close()


def generate_top_cast_graph(movies_actors, output_filename):
    """Generate a graph for the 10 most used cast members."""
    # Count the frequency of each actor
    actor_frequency = {}
    for cast in movies_actors.values():
        for actor in cast:
            if actor not in actor_frequency:
                actor_frequency[actor] = 0
            actor_frequency[actor] += 1

    # Sort and select the top 10 actors
    top_actors = sorted(actor_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
    actors, frequencies = zip(*top_actors)

    # Plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(frequencies), y=list(actors))
    plt.xlabel("Number of Appearances")
    plt.ylabel("Actors")
    plt.title("Top 10 Most Used Cast Members on Netflix")
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()


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


def gen_gephi_id_label_popularity_label_conditional_on_modularity_class(
    actor_to_popularity, actor_to_modularity_class, output_filename
):
    """Generate CSV file for Gephi to read in nodes, but only label top 10 actors per modularity class for modularity classes with 1% or more of the total actors."""
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
