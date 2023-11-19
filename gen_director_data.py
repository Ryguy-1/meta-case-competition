"""Generates all data related to directors."""

import pandas as pd
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

    ##### POPULATE MOVIES DIRECTORS #####
    movies_directors = {}
    for movie, director in zip(netflix_data["title"], netflix_data["director"]):
        movies_directors[movie] = director if isinstance(director, str) else ""

    ##### POPULATE DIRECTOR DETAILS #####
    director_details = {}
    with open("data/col_4_director_data.json", "r") as file:
        director_details = json.load(file)

    ##### POPULATE DIRECTOR TO POPULARITY #####
    director_to_popularity = {}
    for director, details in director_details.items():
        popularity = details.get("popularity", 0)
        director_to_popularity[director] = popularity

    ##### POPULATE DIRECTOR TO AGE #####
    director_to_age = {}
    for director, details in director_details.items():
        birthday = details.get("birthday", None)
        if birthday:
            birthday = datetime.datetime.strptime(birthday, "%Y-%m-%d")
            today = datetime.datetime.today()
            age = (
                today.year
                - birthday.year
                - ((today.month, today.day) < (birthday.month, birthday.day))
            )
            director_to_age[director] = age

    ##### POPULATE DIRECTOR TO GENDER #####
    director_to_gender = {}
    for director, details in director_details.items():
        gender = details.get("gender", 0)
        if gender == 1:
            director_to_gender[director] = "F"
        elif gender == 2:
            director_to_gender[director] = "M"

    ##### GENERATE DIRECTOR DATA GRAPHS OVER TIME #####
    gen_director_popularity_over_time(
        netflix_data=netflix_data,
        director_to_popularity=director_to_popularity,
        output_filename="generated/col_4_popularity_of_director_over_time.png",
    )
    gen_director_age_over_time(
        netflix_data=netflix_data,
        director_to_age=director_to_age,
        output_filename="generated/col_4_age_of_director_over_time.png",
    )
    gen_director_gender_over_time(
        netflix_data=netflix_data,
        director_to_gender=director_to_gender,
        output_filename="generated/col_4_gender_of_director_over_time.png",
    )


def gen_director_popularity_over_time(
    netflix_data, director_to_popularity, output_filename
):
    year_to_director_popularity = {}
    for _, row in netflix_data.iterrows():
        if not isinstance(row["director"], str):
            continue
        if not isinstance(row[DATE_ADDED_COLUMN_NAME], str):
            continue
        date_added = datetime.datetime.strptime(
            row[DATE_ADDED_COLUMN_NAME].strip(), "%B %d, %Y"
        )
        director = row["director"].strip()
        if date_added.year not in year_to_director_popularity:
            year_to_director_popularity[date_added.year] = []
        if director in director_to_popularity:
            year_to_director_popularity[date_added.year].append(
                director_to_popularity[director]
            )

    year_to_avg_director_popularity = {
        year: sum(popularities) / len(popularities) if popularities else 0
        for year, popularities in year_to_director_popularity.items()
    }

    sns.set_theme(style="darkgrid")
    sns.lineplot(
        x=list(year_to_avg_director_popularity.keys()),
        y=list(year_to_avg_director_popularity.values()),
    )
    plt.ylim(bottom=0)
    plt.xlabel("Year")
    plt.ylabel("Average Director Popularity")
    plt.title("Average Director Popularity Over Time")
    plt.savefig(output_filename)
    plt.close()


def gen_director_age_over_time(netflix_data, director_to_age, output_filename):
    year_to_director_age = {}
    for _, row in netflix_data.iterrows():
        if not isinstance(row["director"], str):
            continue
        if not isinstance(row[DATE_ADDED_COLUMN_NAME], str):
            continue
        date_added = datetime.datetime.strptime(
            row[DATE_ADDED_COLUMN_NAME].strip(), "%B %d, %Y"
        )
        director = row["director"].strip()
        if date_added.year not in year_to_director_age:
            year_to_director_age[date_added.year] = []
        if director in director_to_age:
            year_to_director_age[date_added.year].append(director_to_age[director])

    year_to_avg_director_age = {
        year: sum(ages) / len(ages) if ages else 0
        for year, ages in year_to_director_age.items()
    }

    sns.set_theme(style="darkgrid")
    sns.lineplot(
        x=list(year_to_avg_director_age.keys()),
        y=list(year_to_avg_director_age.values()),
    )
    plt.ylim(bottom=0)
    plt.xlabel("Year")
    plt.ylabel("Average Director Age")
    plt.title("Average Director Age Over Time")
    plt.savefig(output_filename)
    plt.close()


def gen_director_gender_over_time(netflix_data, director_to_gender, output_filename):
    year_to_gender_count = {}
    for _, row in netflix_data.iterrows():
        if not isinstance(row["director"], str):
            continue
        if not isinstance(row[DATE_ADDED_COLUMN_NAME], str):
            continue
        date_added = datetime.datetime.strptime(
            row[DATE_ADDED_COLUMN_NAME].strip(), "%B %d, %Y"
        )
        director = row["director"].strip()
        if date_added.year not in year_to_gender_count:
            year_to_gender_count[date_added.year] = {"F": 0, "M": 0}
        if director in director_to_gender:
            director_gender = director_to_gender[director]
            year_to_gender_count[date_added.year][director_gender] += 1

    years = []
    male_counts = []
    female_counts = []
    for year, genders in year_to_gender_count.items():
        years.append(year)
        male_counts.append(genders["M"])
        female_counts.append(genders["F"])

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(10, 6))
    plt.plot(years, male_counts, label="Male")
    plt.plot(years, female_counts, label="Female")
    plt.xlabel("Year")
    plt.ylabel("Count of Directors")
    plt.title("Gender Distribution of Directors Over Time")
    plt.legend()
    plt.savefig(output_filename)
    plt.close()


if __name__ == "__main__":
    main()
