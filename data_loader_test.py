
from data_loader import data_loader

y = data_loader(file_path="conditions_gastrointestinal.csv")
y = data_loader(file_path="conditions_gastrointestinal.csv")
y.download_csv()
y.identify_type()
y.clean_junk()
y.select_year("all")
y.age_death_variable()

