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


if __name__ == "__main__":
    main()
