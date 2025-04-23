import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import os

class Graph:
    """
    Graph class for visualizing variable relationships with lifespan using appropriate plots.
    Automatically selects plot type based on data type suffixes:
        - '__Q' → Quantitative → Scatter plot
        - '__C' → Categorical → Box plot or Bar plot
        - '__D' → Date/Time → Line plot
        - '__I' → Identifiers (Index-like) → Treated as Categorical → Box/Bar plot
    Outputs individual PNGs and compiles a multi-page PDF report.
    """
    def __init__(self, dataframe, folder_name="Results"):
        """
        Initialize the Graph object.
        Args:
            dataframe (pd.DataFrame): Preprocessed DataFrame with suffix-tagged columns.
            folder_name (str): Folder where output plots and PDF report will be saved.
        """
        self.df = dataframe
        self.color_palette = sns.color_palette("colorblind")
        self.pdf_pages = []
        self.output_dir = folder_name
        os.makedirs(self.output_dir, exist_ok=True)

    def save_fig_and_record(self, fig, filename):
        """
        Save a figure to PNG and append it for inclusion in the PDF report.
        """
        filepath = os.path.join(self.output_dir, f"{filename}.png")
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        self.pdf_pages.append(fig)
        print(f"[Saved] {filepath} (added to PDF report)")

    def simple_graph(self, variable, lifespan_column, save=False, filename="simple_plot", bar=False, log_y=False):
        """
        Generate an appropriate plot of a variable against lifespan.

        Plot type is selected based on variable suffix:
            - '__Q' → Scatterplot
            - '__C' or '__I' → Boxplot (or Barplot if bar=True)
            - '__D' → Lineplot over time

        Args:
            variable (str): Feature column name.
            lifespan_column (str): Target column name (e.g., lifespan).
            save (bool): Whether to save the figure as PNG and record for PDF.
            filename (str): Output file name prefix (no extension).
            bar (bool): If True, use barplot for categorical instead of boxplot.
            log_y (bool): If True, use log scale for y-axis.
        """
        if variable not in self.df.columns or lifespan_column not in self.df.columns:
            print(f"[Error] Column not found: '{variable}' or '{lifespan_column}'")
            return

        # Drop rows with missing data
        df_plot = self.df[[variable, lifespan_column]].dropna()
        if df_plot.empty:
            print(f"[Warning] No data to plot for '{variable}' vs '{lifespan_column}'")
            return

        # Handle malformed or partial date strings (e.g., 2022-05-00 → 2022-05-01)
        if variable.endswith("__D"):
            df_plot[variable] = df_plot[variable].astype(str).str.replace(r"-00$", "-01", regex=True)
            df_plot[variable] = pd.to_datetime(df_plot[variable], errors="coerce")
            df_plot = df_plot.dropna()

        plt.clf()
        plt.close()
        fig = plt.figure(figsize=(10, 8))

        clean_var = variable.split("__")[0]
        clean_life = lifespan_column.split("__")[0]

        # Categorical or Identifier → Boxplot (or Barplot if specified)
        if variable.endswith("__C") or variable.endswith("__I"):
            if bar:
                sns.barplot(data=df_plot, x=variable, y=lifespan_column, estimator="mean", errorbar="sd")
            else:
                sns.boxplot(data=df_plot, x=variable, y=lifespan_column)
            plt.xticks(rotation=45, ha="right", fontsize=14)

        # Date → Lineplot over time
        elif variable.endswith("__D"):
            df_plot = df_plot.sort_values(by=variable)
            sns.lineplot(data=df_plot, x=variable, y=lifespan_column, marker="o")

        # Quantitative → Scatterplot
        elif variable.endswith("__Q"):
            sns.scatterplot(data=df_plot, x=variable, y=lifespan_column, color=self.color_palette[0])

        # Fallback for unhandled suffixes
        else:
            print(f"[Warning] Unsupported variable type: {variable}")
            return

        # Log scale for y-axis if enabled
        if log_y:
            plt.yscale("log")

        # Set labels and title
        plt.title(f"{clean_var} vs {clean_life}", fontsize=20, fontweight='bold')
        plt.xlabel(clean_var, fontsize=18, fontweight='bold')
        plt.ylabel(clean_life, fontsize=18, fontweight='bold')
        plt.tight_layout()

        if save:
            self.save_fig_and_record(fig, filename)

        plt.show()
        plt.close()

    def predicted_and_actual(self, actual, predicted, save=False, filename="predicted_vs_actual"):
        """
        Plots predicted lifespan vs actual lifespan as a scatter plot with a diagonal reference line.

        Args:
            actual (str): Name of the column with actual lifespan values.
            predicted (str): Name of the column with predicted lifespan values.
            save (bool): Whether to save the plot.
            filename (str): Filename for the saved plot (no extension).
        """
        if actual not in self.df.columns or predicted not in self.df.columns:
            print(f"[Error] Column not found: '{actual}' or '{predicted}'")
            return

        df_plot = self.df[[actual, predicted]].dropna()
        if df_plot.empty:
            print(f"[Warning] No data to plot for predicted vs actual.")
            return

        plt.clf()
        plt.close()
        fig = plt.figure(figsize=(10, 8))

        sns.scatterplot(data=df_plot, x=actual, y=predicted, color=self.color_palette[1])

        # Reference line for perfect prediction
        min_val = min(df_plot[actual].min(), df_plot[predicted].min())
        max_val = max(df_plot[actual].max(), df_plot[predicted].max())
        plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray", label="Perfect Prediction")

        plt.title("Predicted vs Actual Lifespan", fontsize=20, fontweight='bold')
        plt.xlabel("Actual Lifespan", fontsize=18, fontweight='bold')
        plt.ylabel("Predicted Lifespan", fontsize=18, fontweight='bold')
        plt.legend()
        plt.tight_layout()

        if save:
            self.save_fig_and_record(fig, filename)

        plt.show()
        plt.close()

    def which_graph(self, variable, lifespan_column, save=False, filename=None, bar=False, log_y=False):
        """
        Helper method that determines default filename and calls `simple_graph()`.

        Args:
            variable (str): Feature column name.
            lifespan_column (str): Target column name.
            save (bool): Whether to save the plot.
            filename (str): Custom filename override (optional).
            bar (bool): Whether to use barplot for categorical variables.
            log_y (bool): Whether to apply logarithmic scale to y-axis.
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

    def save_pdf_report(self, path="report.pdf"):
        """
        Save all recorded figures into a multi-page PDF report.

        Args:
            path (str): Output PDF filename (within the results folder).
        """
        if self.pdf_pages:
            report_path = os.path.join(self.output_dir, path)
            with PdfPages(report_path) as pdf:
                for fig in self.pdf_pages:
                    pdf.savefig(fig)
            print(f"\n📄 PDF report saved as: {report_path}")
        else:
            print("\n⚠️ No plots were added to the PDF report.")
