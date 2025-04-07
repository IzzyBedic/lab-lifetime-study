import matplotlib.pyplot as plt
import seaborn as sns

class Graph:
    def __init__(self, dataframe):
        """
        Initializes the Graph class with a preprocessed DataFrame
        from the data_loader module.
        """
        self.df = dataframe
        self.color_palette = sns.color_palette("colorblind")

    def simple_graph(self, variable, lifespan_column, save=False, filename="simple_plot"):
        """
        Plots any input variable against a dog's lifespan.
        Dynamically adjusts figure size and axis limits based on data.
        Supports saving as PNG and PDF.
        """
        if variable not in self.df.columns or lifespan_column not in self.df.columns:
            print(f"[Error] Column not found: '{variable}' or '{lifespan_column}'")
            return

        # Filter out NaNs
        df_plot = self.df[[variable, lifespan_column]].dropna()

        # Dynamic sizing
        height = max(6, min(len(df_plot) / 100, 12))  # Adjust height based on data size
        width = 10  # default width

        if variable.endswith("__C"):
            num_categories = df_plot[variable].nunique()
            width = max(8, min(num_categories * 0.6, 20))
            plt.figure(figsize=(width, height))
            sns.boxplot(data=df_plot, x=variable, y=lifespan_column, palette=self.color_palette)
            plt.xticks(rotation=45, ha='right')

        elif variable.endswith("__Q") or variable.endswith("__D"):
            width = max(8, min(len(df_plot) / 80, 15))  # wider if more data points
            plt.figure(figsize=(width, height))
            sns.scatterplot(data=df_plot, x=variable, y=lifespan_column, palette=self.color_palette)

        else:
            print(f"[Warning] Unsupported variable type: {variable}")
            return

        # Formatting
        plt.title(f"{variable.replace('__', ' ')} vs Lifespan", fontsize=14)
        plt.xlabel(variable.replace('__', ' '), fontsize=12)
        plt.ylabel("Lifespan", fontsize=12)
        plt.tight_layout()

        if save:
            plt.savefig(f"{filename}.png")
            plt.savefig(f"{filename}.pdf")
            print(f"[Saved] {filename}.png and {filename}.pdf")

        plt.show()

    def predicted_and_actual(self, actual, predicted, save=False, filename="predicted_vs_actual"):
        """
        Generates a scatterplot of predicted vs actual lifespan
        with a dashed line indicating perfect prediction.
        Adjusts figure size based on dataset.
        """
        if actual not in self.df.columns or predicted not in self.df.columns:
            print(f"[Error] Column not found: '{actual}' or '{predicted}'")
            return

        df_plot = self.df[[actual, predicted]].dropna()
        height = max(6, min(len(df_plot) / 100, 12))
        width = max(8, min(len(df_plot) / 80, 15))

        plt.figure(figsize=(width, height))
        sns.scatterplot(data=df_plot, x=actual, y=predicted, palette=self.color_palette)
        plt.plot(
            [df_plot[actual].min(), df_plot[actual].max()],
            [df_plot[actual].min(), df_plot[actual].max()],
            linestyle='--', color='gray', label='Perfect Prediction'
        )

        plt.title("Predicted vs Actual Lifespan", fontsize=14)
        plt.xlabel("Actual Lifespan", fontsize=12)
        plt.ylabel("Predicted Lifespan", fontsize=12)
        plt.legend()
        plt.tight_layout()

        if save:
            plt.savefig(f"{filename}.png")
            plt.savefig(f"{filename}.pdf")
            print(f"[Saved] {filename}.png and {filename}.pdf")

        plt.show()

    def which_graph(self, variable, lifespan_column, save=False, filename=None):
        """
        Determines and draws the appropriate graph based on variable type.
        Calls simple_graph with dynamic figure sizing.
        """
        if variable not in self.df.columns:
            print(f"[Error] Variable '{variable}' not found in dataframe.")
            return

        auto_filename = f"{variable}_vs_{lifespan_column}"
        self.simple_graph(variable, lifespan_column, save=save, filename=filename or auto_filename)
