from openai import OpenAI
from data.api_keys import openai as config
from typing import List
import json


def predict_movie_category(movie_descriptions: List[str]) -> str:
    """Predict the category of a list of movie descriptions."""
    client = OpenAI(api_key=config["openai_api_key"])
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": """
                    You are a helpful assistant designed to output JSON. 
                    Your JSON must be valid or the program will crash. 
                    The JSON must have a single root key named 'category' with a string value.
                    You will be given a list of movie descriptions, and you must output the category 
                    they collectively belong to by identifying similarities between them.
                    The category should be a 2-4 word phrase describing the general theme of the movies.
                """,
            },
            {
                "role": "user",
                "content": str(movie_descriptions),
            },
        ],
        temperature=0.0,  # deterministic
    )
    response_json = json.loads(response.choices[0].message.content)
    if "category" not in response_json:
        raise Exception("Invalid response from OpenAI API")
    return response_json["category"]
