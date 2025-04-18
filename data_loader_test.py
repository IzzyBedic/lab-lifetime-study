
from data_loader import data_loader

x = data_loader(file_path="study_endpoints.csv")
x.download_csv()
x.identify_type()
y = data_loader(file_path="conditions_gastrointestinal.csv")
y.download_csv()
y.identify_type()
y.select_year("year5")
y.clean_missing()
y.clean_junk()
print(y.file)

