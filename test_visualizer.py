from GRLT_analysis.data_loader import data_loader
from GRLT_analysis.data_visualizer import Graph
import numpy as np

# Load and process data
loader = data_loader("data/study_endpoints.csv")
loader.download_csv()
loader.identify_type()
loader.select_year("year2")
print(loader.file.columns)

loader_y_axis = data_loader("data/conditions_gastrointestinal.csv")
loader_y_axis.download_csv()
loader_y_axis.identify_type()
loader_y_axis.select_year("year1")

# Simulate target and prediction
# loader.file["lifespan__Q"] = loader.file["tier_of_confidence__Q"] * 2 + np.random.normal(0, 1, size=len(loader.file))
# loader.file["predicted_lifespan__Q"] = loader.file["tier_of_confidence__Q"] * 2 + 0.5

# Simulate target and prediction
loader_y_axis.file["any__predicted"] = loader_y_axis.file["any__C"] * 2 + np.random.normal(0, 1, size=len(loader_y_axis.file))
loader_y_axis.file["predicted_lifespan__C"] = loader_y_axis.file["any__predicted"] * 2 + 0.5

# Visualize
# viz = Graph(loader.file)
# viz.predicted_and_actual("lifespan__Q", "predicted_lifespan__Q")
viz = Graph(loader_y_axis.file)
viz.predicted_and_actual("any__predicted", "predicted_lifespan__C")

# try running saf = True, bar = True, logy = True to try all features
# share plots w/ Madhi