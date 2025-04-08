import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Graph:
    def __init__(self, dataframe):
        """
        Initializes the Graph class with a preprocessed DataFrame.
        """
        self.df = dataframe
        self.color_palette = sns.color_palette("colorblind")

    def simple_graph(
        self,
        variable,
        lifespan_column,
        save=False,
        filename="simple_plot",
        bar=False,
        log_y=False
    ):
        """
        Plots variable vs lifespan with automatic:
        - Boxplot (categorical)
        - Bar chart with error bars (bar=True)
        - Scatterplot (quantitative)
        - Line plot (date)
        Optional:
        - Log scale for y-axis
        - Saving as PNG/PDF
        """
        if variable not in self.df.columns or lifespan_column not in self.df.columns:
            print(f"[Error] Column not found: '{variable}' or '{lifespan_column}'")
            return

        df_plot = self.df[[variable, lifespan_column]].dropna()

        # Convert to datetime if it's a date variable
        if variable.endswith("__D"):
            df_plot[variable] = pd.to_datetime(df_plot[variable], errors="coerce")
            df_plot = df_plot.dropna()

        # Dynamic height based on data size
        height = max(6, min(len(df_plot) / 100, 12))

        if variable.endswith("__C"):
            num_categories = df_plot[variable].nunique()
            width = max(8, min(num_categories * 0.6, 20))
            plt.figure(figsize=(width, height))

            if bar:
                sns.barplot(
                    data=df_plot,
                    x=variable,
                    y=lifespan_column,
                    estimator="mean",
                    ci="sd",  # Standard deviation as error bars
                    palette=self.color_palette
                )
            else:
                sns.boxplot(
                    data=df_plot,
                    x=variable,
                    y=lifespan_column,
                    palette=self.color_palette
                )

            plt.xticks(rotation=45, ha="right")

        elif variable.endswith("__D"):
            width = max(8, min(len(df_plot) / 80, 15))
            plt.figure(figsize=(width, height))
            df_plot = df_plot.sort_values(by=variable)
            sns.lineplot(data=df_plot, x=variable, y=lifespan_column, marker="o")

        elif variable.endswith("__Q"):
            width = max(8, min(len(df_plot) / 80, 15))
            plt.figure(figsize=(width, height))
            sns.scatterplot(
                data=df_plot,
                x=variable,
                y=lifespan_column,
                palette=self.color_palette
            )

        else:
            print(f"[Warning] Unsupported variable type: {variable}")
            return

        if log_y:
            plt.yscale("log")

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
        Scatterplot of predicted vs actual lifespan.
        Adds a perfect-prediction line. Dynamically sized and savable.
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
            linestyle="--", color="gray", label="Perfect Prediction"
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

    def which_graph(
        self,
        variable,
        lifespan_column,
        save=False,
        filename=None,
        bar=False,
        log_y=False
    ):
        """
        Determines appropriate graph type and calls simple_graph().
        Use:
        - bar=True for bar chart (categorical)
        - log_y=True for log-scale lifespan
        """
        if variable not in self.df.columns:
            print(f"[Error] Variable '{variable}' not found in dataframe.")
            return

        auto_filename = f"{variable}_vs_{lifespan_column}"
        self.simple_graph(
            variable,
            lifespan_column,
            save=save,
            filename=filename or auto_filename,
            bar=bar,
            log_y=log_y
        )
