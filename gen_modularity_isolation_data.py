import pandas as pd
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
import json


def main():
    edges_path = "generated/col_4_gephi_cast_edges.csv"
    communities_path = "data/gephi_10_modularities.csv"
    if not os.path.exists("generated/col_4_gephi_cast_edges.csv"):
        print("Please generate the gephi cast edges file first.")
        return
    isolation_metrics = calculate_isolation_metric(
        pd.read_csv(communities_path), pd.read_csv(edges_path)
    )

    ##### Generate Graph of Isolation Metric Per Country #####
    print(isolation_metrics)
    modularity_classes_to_country_names = {
        186: ""
    }
    

def calculate_isolation_metric(modularities_df, edges_df):
    """
    Calculate the isolation metric for each modularity class.

    Args:
    modularities_df (pd.DataFrame): DataFrame containing node IDs and their corresponding modularity classes.
    edges_df (pd.DataFrame): DataFrame containing the edges between nodes.

    Returns:
    dict: A dictionary where keys are modularity classes and values are their isolation metrics.
    """

    # Mapping each node to its modularity class
    node_to_class = dict(
        zip(modularities_df["Id"], modularities_df["modularity_class"])
    )

    # Initializing counters for internal and external edges for each class
    internal_edges = {
        mod_class: 0 for mod_class in modularities_df["modularity_class"].unique()
    }
    external_edges = {
        mod_class: 0 for mod_class in modularities_df["modularity_class"].unique()
    }

    # Counting internal and external edges
    for _, edge in edges_df.iterrows():
        source_class = node_to_class.get(edge["Source"])
        target_class = node_to_class.get(edge["Target"])

        if source_class is not None and target_class is not None:
            if source_class == target_class:
                internal_edges[source_class] += 1
            else:
                external_edges[source_class] += 1
                external_edges[target_class] += 1

    # Calculating isolation metric for each modularity class
    isolation_metric = {}
    for mod_class in internal_edges:
        total_edges = internal_edges[mod_class] + external_edges[mod_class]
        if total_edges > 0:
            isolation_metric[mod_class] = internal_edges[mod_class] / total_edges
        else:
            isolation_metric[mod_class] = float(
                "nan"
            )  # No edges connected to this class

    return dict(
        sorted(
            isolation_metric.items(),
            key=lambda item: (math.isnan(item[1]), item[1]),
            reverse=True,
        )
    )


if __name__ == "__main__":
    main()
