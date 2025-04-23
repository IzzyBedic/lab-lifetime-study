from GRLT_analysis.data_loader import data_loader
import pandas as pd

# Load and process data
y = data_loader(file_path="data/conditions_gastrointestinal.csv")
y.download_csv()
y.identify_type()
y.select_year("year5")
y.clean_junk()

loader = data_loader("data/study_endpoints.csv")
loader.download_csv()
loader.identify_type()

new_df = y.file
new_df["outcome"] = loader.file["subject_id__I"] # if the dog died, it's id would be in the subject_id__I column.
                                                 # NaN == dead dog
print(new_df["outcome"])
new_df["outcome"] = [pd.isna(x) is False for x in new_df["outcome"]]  # then, set "outcome" to a categorical True or False where True is that the dog is alive
new_df["outcome"] = new_df["outcome"].fillna(False)
print(new_df["outcome"])

#y = new_df["outcome"]
#X = new_df[]


"""forward_subset_selection(X, y,
                         val_ratio=0.2,
                         epsilon=1e-4,
                         max_features=10,
                         verbose=True,
                         plot=True,
                         scale=True)"""