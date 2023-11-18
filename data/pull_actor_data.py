import pandas as pd
import os
import json
import concurrent.futures
import time
from typing import Dict
import requests
from api_keys import tmdb as config

OUTPUT_FILENAME = "col_4_actors_data.json"


def load_current_actor_data() -> Dict:
    """Load current actor data from file."""
    if not os.path.exists(OUTPUT_FILENAME):
        return {}
    with open(OUTPUT_FILENAME, "r") as file:
        return json.load(file)


def save_current_actor_data(stats: Dict):
    """Save current actor data to file."""
    with open(OUTPUT_FILENAME, "w") as file:
        json.dump(stats, file, indent=4)


def get_actor_details(actor_name: str) -> Dict:
    """Get actor details from TMDB API."""
    api_key = config["tmdb_api_key"]
    search_url = f"https://api.themoviedb.org/3/search/person?api_key={api_key}&query={actor_name}"
    response = requests.get(search_url).json()
    if response["results"]:
        actor_id = response["results"][0]["id"]
        details_url = (
            f"https://api.themoviedb.org/3/person/{actor_id}?api_key={api_key}"
        )
        details_response = requests.get(details_url).json()
        return details_response
    else:
        return {}


if __name__ == "__main__":
    data = pd.read_csv("netflix_titles.csv")
    movies_actors = {}
    for movie, actors in zip(data["title"], data["cast"]):
        if isinstance(actors, str):
            movies_actors[movie] = [x.strip() for x in actors.split(",")]
            continue
        movies_actors[movie] = []

    # Get Each Actor's Stats
    all_actors = list(set([actor for cast in movies_actors.values() for actor in cast]))
    futures = {}
    actor_details = {}

    # Load Actor Data from File (initialize)
    actor_details = load_current_actor_data()

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
