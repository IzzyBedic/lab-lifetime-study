from forward_selection import forward_subset_selection
from data_loader import data_loader
import pandas as pd

# Load and process data
y = data_loader(file_path="conditions_gastrointestinal.csv")
y.download_csv()
y.identify_type()
y.select_year("year5")
y.clean_junk()

loader = data_loader("study_endpoints.csv")
loader.download_csv()
loader.identify_type()

new_df = y.file
new_df["outcome"] = loader.file["subject_id__I"] # if the dog died, it's id would be in the subject_id__I column.
                                                # NaN == dead dog
new_df["outcome"] = [x is not None for x in new_df["outcome"]]     # then, set "outcome" to a categorical 1 or 0 where 1 is that the dog is alive
new_df["outcome"] = new_df["outcome"].fillna(False)
print(new_df[new_df["outcome" == False]].tail(100))

#X = new_df.drop(["outcome"])
#y = new_df["outcome"].values

"""forward_subset_selection(X, y,
                         val_ratio=0.2,
                         epsilon=1e-4,
                         max_features=10,
                         verbose=True,
                         plot=True,
                         scale=True)"""