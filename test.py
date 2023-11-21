import pandas as pd
import plotly_express as px

df = pd.read_csv("test.csv")

############################ CREATE ANIMATED PLOT ############################
# Animated Visualization for individual points
fig = px.scatter(
    df,
    x="x",
    y="y",
    animation_frame="month_year_added",  # Updated to reflect month-year
    color="categories",
    hover_name="movie_titles",
    hover_data=["categories"],
    title="Evolution of Netflix Global Catalog",
    labels={"month_year_added": "Date Added to Netflix"},  # Updated label
    # size_max=100,  # Adjust the maximum size of bubbles
)

fig.update_layout(showlegend=True)
fig.show()

# Optional: Save the animation as HTML or capture it as a video/gif
fig.write_html("test_gen.html")  # Updated file name
