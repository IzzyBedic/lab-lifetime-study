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
    def __init__(self, file_path):
        self.file_path = file_path

    def download_csv(self):
        """
        Uses the file name to download the file from a local device.
        Double check that the file path is correct
        :return: the downloaded file
        """
        download = pd.read_csv(self.file_path)
        return(download)

    def identify_type(self):
        """
        Categorizes variables as categorical or quantitative to facilitate regression analysis
        Updates column names with __C or __Q
        :return: None
        """
        column_names = pd.DataFrame.columns
        print(column_names)


    def clean_missing(self):
        """
        handles missing data
        :return:
        """
        pass

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
test = x.download_csv()
x.identify_type()

print(test)


