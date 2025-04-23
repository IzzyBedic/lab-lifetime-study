**READ ME**
<br>
Hello! This is the lab lifetime study package. It is designed to facilitate intuitive and easy EDA of data from the Golden Lifetime Retriver Study at the Morris Animal
Foundation. 

<br>
Within this repository there are 3 types of files:
- The __init__ file
- The class code for data_loader, data_visualizer, and data_analysis
- For each class document there is a _test file where test cases for that class

**Where might this come in handy?**
<br>
This package focuses on the data relating to health outcomes for golden retreivers. This is meant to quickly acertain what variables in a given file would be able to 
be quickly fed into a graph for EDA. Once EDA is done, a user could then use the variables they were interested in using the analysis class to run a regression using whether a given golden retriever is alive or not.
<br>

*Example:*
You find a file you really want to look into. You think gastrointentinal issues could be a major contributor to the early death of golden retrievers. You run variable_name = data_loader(file_path="file_name.csv") to get the file with gastrointestinal-related data into the function. Then you run all of the functions, variable_name.download_csv() turns the csv file into a pandas dataframe, variable_name.identify_type() marks columns by whether they are categorical, quantitative, date, or an ID column.
variable_name.clean_missing() marks columns with lots of missingness. variable_name.clean_junk() gets rid of columns marked by clean_missing() and also by how varaible the data is. If more than 10% of the data is made up of completely different entries (with the exception of the ID column), then the column is determined to be too messy and excluded. variable_name.select_year("year_num" or "all") helps a user narrow down the timepoint which they are looking into. If they wanted to look at year 5 for example, they'd enter ("year5"), if they wanted an average across all years they'd enter ("all"). Then, to finish, you'd run variable_name.age_death_variable(required_file = "data/dog_profile.csv"). This would create a "lifespan" column based off of birth and death dates in dog_profile.csv (obtained from the Bette M Morris website). It merges the lifespan column with your dataset so that you can visualize and analyze variables against lifespan. 
<br>

So, then you want to visually explore the data. You'd first use variable_name.simple_graph(self, variable, lifespan_column), which would return a scatterplot if the column was quantitative (ends in '__Q' as per .identify_type()), a boxplot if categorical or the ID column, and a lineplot if a date column. variable_name.which_graph(variable, lifespan_column) is a helper function for simple_graph. Another way to graphically represent your data is with variable_name.predicted_and_actual(), which shows a graph of predicted lifespan vs actual lifespan depending on the input variables. Then, to save your graphs, you can use variable_name.save_fig_and_record and variable_name.save_pdf_report().
<br>

For analysis, you need a NumPy array of data, and once you input it you get back various models run on the variables given. variable_name.forward_subset_selection() and variable_name.lasso_feature_selection() are the two main models used. Both models do similar modelling, but approached differently. Running both and comparing the results can help make sure that a user's results are accurate.
