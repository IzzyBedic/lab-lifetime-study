from math import isnan

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
            if "count" in list:
                quantitative_name = column_names[i] + "__Q"
                self.file = self.file.rename(columns={column_names[i]: quantitative_name})
            elif ("is" == list[0]) or ("any" == list[0]) or ("to_date" == column_names[i]):
                categorical_name = column_names[i] + "__C"
                self.file = self.file.rename(columns = {column_names[i]: categorical_name})
            elif column_names[i] == "subject_id":
                id_name = column_names[i] + "__I"
                self.file = self.file.rename(columns={column_names[i]: id_name})
            elif any([x in date_keywords for x in list]):
                date_name = column_names[i] + "__D"
                self.file = self.file.rename(columns={column_names[i]: date_name})
            elif self.file[column_names[i]].dtype == "int64":
                quantitative_name = column_names[i] + "__Q"
                self.file = self.file.rename(columns={column_names[i]: quantitative_name})
            else:
                categorical_name = column_names[i] + "__C"
                self.file = self.file.rename(columns={column_names[i]: categorical_name})

    def select_year(self, year_num):
        """
        Updates the dataset to only include the data within the desired time frame, either one year or all.
        :param year_num: The year the user wants to select. Options are "all", "year1", "year2", etc.
        :return: None
        """
        integer_search = r"\d+"
        if "year_in_study" in self.file.columns:
            print("year in study is not an included variable in this dataset")
        elif year_num != "all" and len(year_num) != 5:
            print("year in study must be input in the format 'year#' or if you want an average of all years, 'all'")
        elif year_num == "all":
            pass
        else:
            sorted_by_year_data = self.file[self.file["year_in_study__D"] == int(year_num[4])]
            self.file = sorted_by_year_data # "all", "most recent", "year1", get rid of dates besides year in study, collapse all year values into an average across categorical


    def clean_missing(self):
        """
        Handles missing data by marking columns that may not be helpful due to an excess of missing data.
        :return: None
        """
        column_names = self.file.columns[1:]
        number_dogs = self.file["subject_id__I"].nunique()
        print("For context, there are", len(self.file)/number_dogs, "entries in this dataset per unique dog for the timeframe you have selected")
        for i in column_names:
            non_na_per_series_item = len(self.file[self.file[i].isna() == True])/len(self.file)
            if non_na_per_series_item >= .25:
                marked_name = i + "__cleanme"
                self.file = self.file.rename(columns={i: marked_name})
            else:
                pass

    def clean_junk(self): # uses clean_missing
        """
        Removes columns containing inconsistently formatted
        or low-value data (to regression analysis or visualization)
        """
        column_names = self.file.columns
        cleaned_df = pd.DataFrame()
        for i in column_names:
            if (i != "subject_id__I" and self.file[i].nunique() >= 1/10*len(self.file[i])) or ("__cleanme" in i):
                pass
            else:
                cleaned_df[i] = self.file[i]
        self.file = cleaned_df


