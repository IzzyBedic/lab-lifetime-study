 Visualization Module Summary
The Graph class developed for this project successfully delivers a modular, automated, and dataset-agnostic visualization pipeline that fully satisfies the objectives outlined in Part 2 of the proposal.

Key features include:

Automatic Plot Selection: The simple_graph() method intelligently selects the appropriate plot type—boxplot, bar chart, scatterplot, or line plot—based on variable type suffixes (__C, __Q, __D, __I), ensuring consistent and meaningful visualizations across all variable types.

Clean, Readable Formatting: All plots are styled using a colorblind-friendly palette, with bold and enlarged axis labels and titles for enhanced readability and publication-quality presentation.

Error-Resilient Plotting: The module robustly handles missing or malformed data (e.g., invalid date formats such as -00) using intelligent preprocessing and dropna() safeguards to avoid runtime errors and generate clean visuals.

Exclusion of Redundant Visuals: The module automatically skips plots that are unlikely to provide insight, such as identity comparisons (lifespan vs lifespan) and ID-based variables (__I), maintaining the relevance and clarity of the visual output.

Prediction Diagnostics: The predicted_and_actual() method visualizes model predictions against actual lifespan values, including a perfect-prediction reference line to assess regression performance.

Structured Output: All generated plots are saved as individual high-resolution PNG files and compiled into a multi-page PDF report. Output is organized into folders named after the source dataset, enabling seamless comparison and traceability across multiple datasets.

This modular framework is fully reusable and scalable for any dataset that follows the established variable suffix convention, making it an effective tool for exploratory data analysis and model evaluation.
