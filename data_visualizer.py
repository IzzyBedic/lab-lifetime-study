import matplotlib.pyplot as plt
import seaborn as sns

class Graph:
    def __init__(self, dataframe):
        """
        Initializes the Graph class with a preprocessed DataFrame
        """
        self.df = dataframe
        self.color_palette = sns.color_palette("colorblind")

    def simple_graph(self, variable, lifespan_column):
        """
        Plots any variable against the dog's lifespan.
        Automatically selects the appropriate plot based on variable type.
        """
        if variable not in self.df.columns or lifespan_column not in self.df.columns:
            print(f"[Error] One or both columns '{variable}', '{lifespan_column}' not found in the dataframe.")
            return

        if variable.endswith("__Q") or variable.endswith("__D"):
            # Quantitative or date-like variable => scatterplot
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=self.df, x=variable, y=lifespan_column, palette=self.color_palette)
        elif variable.endswith("__C"):
            # Categorical => boxplot
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=self.df, x=variable, y=lifespan_column, palette=self.color_palette)
            plt.xticks(rotation=45, ha='right')

        plt.title(f"{variable.replace('__', ' ')} vs Lifespan")
        plt.xlabel(variable.replace('__', ' '))
        plt.ylabel("Lifespan")
        plt.tight_layout()
        plt.show()

    def predicted_and_actual(self, actual, predicted):
        """
        Overlays predicted vs actual lifespan in a scatterplot.
        """
        if actual not in self.df.columns or predicted not in self.df.columns:
            print(f"[Error] Columns '{actual}' or '{predicted}' not found in dataframe.")
            return

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=self.df, x=actual, y=predicted, palette=self.color_palette)
        plt.plot([self.df[actual].min(), self.df[actual].max()],
                 [self.df[actual].min(), self.df[actual].max()],
                 linestyle='--', color='gray', label='Perfect Prediction')
        plt.xlabel("Actual Lifespan")
        plt.ylabel("Predicted Lifespan")
        plt.title("Predicted vs Actual Lifespan")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def which_graph(self, variable, lifespan_column):
        """
        Determines and calls the correct plot type based on variable's suffix.
        """
        if variable.endswith("__Q") or variable.endswith("__D"):
            self.simple_graph(variable, lifespan_column)
        elif variable.endswith("__C"):
            self.simple_graph(variable, lifespan_column)
        else:
            print(f"[Info] Skipping unsupported variable: {variable}")

