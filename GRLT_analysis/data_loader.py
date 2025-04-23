from math import isnan
from unicodedata import numeric
import pandas as pd
import re

class data_loader:
    """
    Data Loading and Filtering Module (data_loader.py):
    This module will utilize Pandas' CSV reader to import
    the dataset and apply custom functions for data
    preprocessing. It will include identify_type, which
    categorizes variables as categorical or quantitative to
    facilitate regression analysis.
    """
    def __init__(self, file_path, file = None):
        self.file_path = file_path
        self.file = None

    def download_csv(self):
        """
        Uses the file name to download the file from a local device.
        Double check that the file path is correct
        :return: the downloaded file
        """
        download = pd.read_csv(self.file_path)
        self.file = download

    def identify_type(self):
        """
        Categorizes variables as id, categorical, or quantitative to facilitate regression analysis
        Updates column names with __I, __C, __D, or __Q
        :return: None
        """
        date_keywords = ["year", "month", "day", "hours", "date"]
        column_names = self.file.columns
        for i in range(0, len(column_names)):
            list = column_names[i].split("_")
            if ("count" in list) or column_names[i] == "year_in_study":
                quantitative_name = column_names[i] + "__Q"
                self.file = self.file.rename(columns={column_names[i]: quantitative_name})
            elif ("is" == list[0]) or ("any" == list[0]) or ("to_date" == column_names[i]):
                categorical_name = column_names[i] + "__C"
                self.file = self.file.rename(columns = {column_names[i]: categorical_name})
            elif column_names[i] == "subject_id":
                id_name = column_names[i] + "__I"
                self.file = self.file.rename(columns={column_names[i]: id_name})
            elif any([x in date_keywords for x in list]) and (self.file[column_names[i]].dtype != "object"): # if a date object and also not a list of dates/not in a strict string or int format
                print(column_names[i])
                date_name = column_names[i] + "__D"
                self.file[column_names[i]] = pd.to_datetime(self.file[column_names[i]]) # convert the date to to_datetime
                self.file = self.file.rename(columns={column_names[i]: date_name})
            elif self.file[column_names[i]].dtype == "int64":
                quantitative_name = column_names[i] + "__Q"
                self.file = self.file.rename(columns={column_names[i]: quantitative_name})
            else:
                categorical_name = column_names[i] + "__C"
                self.file = self.file.rename(columns={column_names[i]: categorical_name})


    def clean_missing(self):
        """
        Handles missing data by marking columns that may not be helpful due to an excess of missing data.
        :return: None
        """
        column_names = self.file.columns[1:]
        number_dogs = self.file["subject_id__I"].nunique()
        print("For context, there are", len(self.file)/number_dogs, "entries in this dataset per unique dog for the timeframe you have selected")
        for i in column_names:
            na_per_series_item = len(self.file[self.file[i].isna() == True])/len(self.file)
            if na_per_series_item >= .25:
                marked_name = i + "__cleanme"
                self.file = self.file.rename(columns={i: marked_name})
            else:
                pass

    def clean_junk(self): # uses clean_missing
        """
        Removes columns containing inconsistently formatted
        or low-value data (to regression analysis or visualization)
        """
        self.clean_missing()
        column_names = self.file.columns
        cleaned_df = pd.DataFrame()
        for i in column_names:
            if (i != "subject_id__I" and self.file[i].nunique() >= 1/10*len(self.file[i])) or ("__cleanme" in i):
                pass
            else:
                cleaned_df[i] = self.file[i]
        self.file = cleaned_df

    def select_year(self, year_num, type_avg_for_categorical = "most_common"):
        """
        !If running into errors with "all", make sure you run "clean_junk" beforehand to get rid of potentially
        empty columns which could mess with the averaging of all columns.

        Updates the dataset to only include the data within the desired time frame, either one year or all.
        :param year_num: The year the user wants to select. Options are "all", "year1", "year2", etc.
        :return: None
        """
        self.type_avg_for_categorical = type_avg_for_categorical
        integer_search = r"\d+"
        if "year_in_study__Q" not in self.file.columns:
            print("year in study is not an included variable in this dataset, this dataset is not valid to use for this function")
        elif year_num != "all" and len(year_num) != 5:
            print("year in study must be input in the format 'year#' or if you want an average of all years, 'all'")
        elif year_num == "all":

            # I needed to reference this article to fix a pesky error https://stackoverflow.com/questions/73601386/why-i-cant-drop-nan-values-with-dropna-function-in-pandas
            self.file = self.file.replace("NaN", pd.NA)

            # dividing up the type of aggregation by variable type
            quantitative_columns = [column for column in self.file.columns if column.endswith("__Q")]
            categorical_columns = [column for column in self.file.columns if column.endswith("__C")]
            categorical_int_columns = [column for column in self.file.columns if (column.endswith("__C") and (self.file[column].dtype == "int64" or self.file[column].dtype == "float64"))]
            date_columns = [column for column in self.file.columns if column.endswith("__D")]

            # quantitative measurements are just averaged
            if len(quantitative_columns) == 0:
                quantitative_df = self.file["subject_id__I"].copy()
            else:
                quantitative_df = self.file.groupby("subject_id__I")[quantitative_columns].mean().reset_index()

            # date measurements are always by most recent
            if len(date_columns) == 0:
                date_df = self.file["subject_id__I"].copy().drop_duplicates()
            else:
                date_columns_full = ["subject_id__I"] + date_columns
                for i in date_columns:
                    sorted_dates = self.file[date_columns_full].sort_values(by=["subject_id__I", i], ascending=[True, False])
                date_df = sorted_dates.drop_duplicates(subset="subject_id__I", keep="first")

            ## categorical measurements can be "averaged" by: most recent input, most common input, or if a value of more
            ## than 0 is *ever* input (for numeric categorical variables)
            if len(categorical_columns) == 0:
                categorical_df = self.file["subject_id__I"].copy()
            else:

                relevant_columns = ["subject_id__I", "year_in_study__Q"] + categorical_columns

                if type_avg_for_categorical == "most_recent":
                    # most recent
                    sorted_file_categorical_for_recent = self.file[relevant_columns].sort_values(by=["subject_id__I", "year_in_study__Q"],ascending=[True, False])
                    most_recent_categorical = sorted_file_categorical_for_recent.drop_duplicates(subset="subject_id__I", keep="first")
                    categorical_df = most_recent_categorical

                elif type_avg_for_categorical == "most_common" or "ever_yes":
                    # I'm putting these together because there's no way to equate a string like "DOG" to a yes or no,
                    # so for string categoricals, I will be defaulting the most common string in a column

                    # most common
                    most_common_values_df = self.file[relevant_columns].copy()
                    for index in range(1, len(relevant_columns)):
                        most_common_values_df[relevant_columns[index]] = self.file.groupby("subject_id__I")[relevant_columns[index]].transform(lambda n: n.mode(dropna = True)[0]) # Gets the most common value for each id
                    most_common_values_df = most_common_values_df.drop_duplicates(subset = "subject_id__I", keep = "first")

                    # ever yes
                    if type_avg_for_categorical == "ever_yes" and len(categorical_int_columns) != 0:
                        # If ever "yes" (or 1+)
                        categorical_int_columns = ["subject_id__I"] + categorical_int_columns
                        categorical_df = self.file[categorical_int_columns].copy()
                        for i in range(1, len(categorical_int_columns)):
                            sorted_by_one = self.file[[categorical_int_columns[i], "subject_id__I"]].sort_values(by=["subject_id__I", categorical_int_columns[i]], ascending=[True, False])
                            ever_yes_categorical = sorted_by_one.drop_duplicates(subset="subject_id__I", keep="first")
                        categorical_df = ever_yes_categorical.join(most_common_values_df, by = "subject_id__I", how = "left").reset_index()
                    else:
                        categorical_df = most_common_values_df.reset_index()

            # merging cat and quant and date by subject_id__I
            merged_df_1 = quantitative_df.merge(categorical_df, on = "subject_id__I", how = "left")
            merged_df_2 = merged_df_1.merge(date_df, on = "subject_id__I", how = "left")
            self.file = merged_df_2

        else:
            sorted_by_year_data = self.file[self.file["year_in_study__Q"] == int(year_num[4])]
            self.file = sorted_by_year_data # "all", "most recent", "year1", get rid of dates besides year in study, collapse all year values into an average across categorical

    def age_death_variable(self, required_file = "data/dog_profile.csv"):
        """
        includes the age of death column in whatever dataframe is being used to run the regression
        on what variables influence age of death
        :param required_file: the file path for the age of death variable used for the regression analysis
        updates to: a dataset consisting of a new column `age of death` merged with the current dataset
        """
        self.required_file = required_file
        loaded_data = pd.read_csv(required_file) # loading the data from dog_profile.csv
        lifespan = pd.DataFrame() # empty dataframe
        passed = loaded_data[loaded_data["study_status"] == "Enrolled Deceased"].copy() # getting all dogs that have passed away in a copied (to remove error) dataframe
        passed["death_date_dt"] = pd.to_datetime(passed["death_date"], format = "%Y-%m") # converting death and birth
        passed["birth_date_dt"] = pd.to_datetime(passed["birth_date"], format = "%Y-%m") # dates to datetime so I can subtract them
        lifespan["lifespan_days__Q"] = passed["death_date_dt"] - passed["birth_date_dt"] # creating a series that is lifespan in days
        lifespan["subject_id__I"] = passed["subject_id"]
        merged_df = self.file.merge(lifespan, how = "left", on = "subject_id__I")
        self.file = merged_df



