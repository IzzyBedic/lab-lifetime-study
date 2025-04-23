import numpy as np
import os
from data_loader import data_loader
from data_visualizer import Graph

# Set the CSV file to load
csv_path = "study_endpoints_with_lifespan.csv"

# Extract folder name from CSV filename
folder_name = os.path.splitext(os.path.basename(csv_path))[0]

# Target Y-axis variable
TARGET = "lifespan__Q"

# Step 1: Load and preprocess data
loader = data_loader(csv_path)
loader.download_csv()

# ✅ Fix for missing suffix on 'lifespan'
if "lifespan" in loader.file.columns and TARGET not in loader.file.columns:
    loader.file = loader.file.rename(columns={"lifespan": TARGET})
    print(f"🔁 Renamed 'lifespan' to '{TARGET}'.")

print("📋 Columns before identify_type():", list(loader.file.columns))
loader.identify_type()

# 🛠 Fix for accidental double renaming like 'lifespan__Q__C'
for col in loader.file.columns:
    if "lifespan" in col and "__Q" in col and col != TARGET:
        loader.file = loader.file.rename(columns={col: TARGET})
        print(f"🔁 Renamed misidentified column '{col}' back to '{TARGET}'")
        break

loader.select_year("year1")
print("📋 Columns after select_year():", list(loader.file.columns))

# # Step 2: Simulate lifespan__Q if missing or all values are NaN
# if TARGET not in loader.file.columns or loader.file[TARGET].isnull().all():
#     loader.file[TARGET] = np.random.normal(10, 2, size=len(loader.file))
#     print(f"[✓] Simulated '{TARGET}'.")

# Step 3: Simulate predicted lifespan for regression plot
loader.file["predicted_lifespan__Q"] = loader.file[TARGET] + np.random.normal(0, 1, size=len(loader.file))

# Step 4: Create visualizer with dynamic output folder
viz = Graph(loader.file, folder_name=folder_name)

# Step 5: Group variables by suffix type
columns_by_type = {"__C": [], "__Q": [], "__D": [], "__I": []}
for col in loader.file.columns:
    for suffix in columns_by_type:
        if col.endswith(suffix):
            columns_by_type[suffix].append(col)

# Step 6: Plot each variable type
for suffix, columns in columns_by_type.items():
    print(f"\n🔍 Plotting variables with suffix '{suffix}':")
    for var in columns:
        # 🚫 Skip unhelpful or redundant plots
        if var == TARGET or var.endswith("__I") or var.startswith("lifespan"):
            continue

        valid_rows = loader.file[[var, TARGET]].dropna().shape[0]
        n_unique = loader.file[var].nunique()

        if valid_rows == 0 or n_unique < 2:
            continue

        if suffix in ["__C", "__I"] and n_unique > 20:
            print(f"🚫 Skipping '{var}' — too many unique categories ({n_unique})")
            continue

        print(f"📊 Plotting '{var}' vs '{TARGET}'...")
        if suffix in ["__C", "__I"]:
            loader.file[var] = loader.file[var].astype("category")

        viz.which_graph(var, TARGET, save=True)

# Step 7: Plot predicted vs actual
print("\n📈 Plotting predicted vs actual lifespan...")
viz.predicted_and_actual(TARGET, "predicted_lifespan__Q", save=True)

# Step 8: Save all plots to one PDF report
viz.save_pdf_report("report.pdf")
print(f"\n✅ All plots saved in '{folder_name}/' and compiled into report.pdf.")