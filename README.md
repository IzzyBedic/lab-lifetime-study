**READ ME**
Hello! This is the lab lifetime study package. It is designed to facilitate intuitive and easy EDA of data from the Golden Lifetime Retriver Study at the Morris Animal
Foundation. 

Within this repository there are 3 types of files:
- The __init__ file
- The class code for data_loader, data_visualizer, and data_analysis
- For each class document there is a _test file where test cases for that class

**Where might this come in handy?**
This package focuses on the data relating to health outcomes for golden retreivers. This is meant to quickly acertain what variables in a given file would be able to 
be quickly fed into a graph for EDA. Once EDA is done, a user could then use the variables they were interested in using the analysis class to run a regression using whether a 
given golden retriever is alive or not.

*Example:*
You find a file you really want to look into. You think gastrointentinal issues could be a major contributor to the early death of golden retrievers. You run x = data_loader(file_path="file_name.csv")
to get the 
