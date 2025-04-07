# test_visualizer.py

from data_loader import data_loader
from visualizer import Graph

# Step 1: Load and preprocess the data
loader = data_loader("study_endpoints.csv")
loader.download_csv()
loader.identify_type()

# Step 2: Initialize the Graph object with the cleaned data
viz = Graph(loader.file)

# Step 3: Try plotting one of the variables against a quantitative lifespan-like column
# Since our dataset lacks a true "lifespan", I'll simulate it using year_in_study__Q

viz.which_graph("tier_of_confidence__Q", "year_in_study__Q")  # Example scatter
viz.which_graph("status__C", "year_in_study__Q")              # Example boxplot

