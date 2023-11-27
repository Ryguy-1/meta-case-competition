# Meta Data Science Case Competition 2023

This is a quick showcase of a few things we were able to generate for the Meta Data Science Case Competition 2023 at UVA.
The contest was a week long, and we were given a historical dataset of the Netflix catalog.
We ended up using lots of outside data (TMDB, IMDB, Wikipedia, etc.) to generate our final suggestions that were on a 10 slide presentation.

```
Note: This was only my part of the project. Other Graphs and Visualizations were made by my teammates as well for the final presentation.
```

## Network Graph of Netflix Talent (Connected by Worked With on a Project)

![Network Graph of Netflix Talent](data/actor-node-map-5k-5k.png)

### Node Size is Proportional to TMDB Popularity. Color is auto-identified by Gephi's community detection algorithm.

![Isolation Index of Netflix Talent](generated/isolation_metric_for_each_color.png)
![Most Popular Country Per Color](generated/most_common_country_for_each_color.png)

## Animated (Open [HTML](generated/cluster_individual_month_year_global.html) File to See) Map of Titles Added Semantically Clustered Through K-Means and Labeled by GPT with Descriptions.

![Animated Map of Titles Added Semantically Clustered Through K-Means and Labeled by GPT with Descriptions](generated/catalog_visualization_screenshot.png)

## Number of Titles Per Production Country Over Time

![Number of Titles Per Production Country Over Time](generated/title_counts_by_country.png)

## Misc Graphs Not Presented in Final Presentation

![G1](generated/col_4_age_of_actor_over_time.png)
![G2](generated/col_4_age_of_director_over_time.png)
![G3](generated/col_4_gender_of_actor_over_time.png)
![G4](generated/col_4_gender_of_director_over_time.png)
![G5](generated/col_4_popularity_of_actor_over_time.png)
![G6](generated/col_4_popularity_of_director_over_time.png)
