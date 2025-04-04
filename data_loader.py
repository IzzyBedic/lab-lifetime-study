import pandas as pd
import re

from pandas import concat
from pandas.core.dtypes.common import is_int64_dtype


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
        Updates column names with __I, __C or __Q
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
        print(self.file.columns)

    def clean_missing(self):
        """
        Handles missing data by marking columns that may not be helpful due to an excess of missing data.
        :return: None
        """



    def clean_junk(self):
        """
        will remove columns containing inconsistently formatted
        or low-value data"""
        pass

    def get_info(self):
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
print()


