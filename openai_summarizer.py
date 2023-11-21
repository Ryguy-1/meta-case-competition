from openai import OpenAI
from data.api_keys import openai as config
from typing import List
import json

client = OpenAI(api_key=config["openai_api_key"])


def predict_movie_category(movie_descriptions: List[str]) -> str:
    """Predict the category of a list of movie descriptions."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {
                "role": "system",
                "content": """
                    You are a helpful assistant designed to output a single category and no more.
                    You are part of a code pipeline, and any output will be used downstream as written.
                    You will be given a list of movie descriptions, and you must output the category that these descriptions collectively belong to by identifying similarities between them.
                    You must output the category and only the category.
                    The category should be a short phrase describing the general theme of the movies (similar to a genre).
                """,
            },
            {
                "role": "user",
                "content": str(movie_descriptions),
            },
        ],
        temperature=0.2,  # more deterministic
    )
    return response.choices[0].message.content
