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

    def clean_missing(self):
        """
        Handles missing data by marking columns that may not be helpful due to an excess of missing data.
        :return: None
        """
        column_names = self.file.columns
        for i in column_names:
            print(i)
            is_there_a_non_na_value_per_subject = self.file.groupby("subject_id__I")[i]
            print(is_there_a_non_na_value_per_subject)
            na_proportion_per_subject = is_there_a_non_na_value_per_subject/self.file.groupby("subject_id__I")[i].count
            if na_proportion_per_subject >= .25:
                marked_name = i + "__cleanme"
                self.file = self.file.rename(columns={i: marked_name})

    def clean_junk(self): # uses clean_missing
        """
        will remove columns containing inconsistently formatted
        or low-value data
        """


    def get_info(self): # hardest!
        """
        retrieves variable descriptions from the
        Betty M. Morris website to enhance interpretability
        :return:
        """
        pass



x = data_loader(file_path="study_endpoints.csv")
x.download_csv()
x.identify_type()
y = data_loader(file_path="conditions_gastrointestinal.csv")
y.download_csv()
y.identify_type()
y.clean_missing()
print()

# filter something or not, set up of threshold for what to do with dogs not present in 7 years
# consider time more


