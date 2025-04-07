from data_loader import data_loader
from data_visualizer import Graph
import numpy as np

# Load and process data
loader = data_loader("study_endpoints.csv")
loader.download_csv()
loader.identify_type()

# Simulate target and prediction
loader.file["lifespan__Q"] = loader.file["tier_of_confidence__Q"] * 2 + np.random.normal(0, 1, size=len(loader.file))
loader.file["predicted_lifespan__Q"] = loader.file["tier_of_confidence__Q"] * 2 + 0.5

# Visualize
viz = Graph(loader.file)
viz.predicted_and_actual("lifespan__Q", "predicted_lifespan__Q")
