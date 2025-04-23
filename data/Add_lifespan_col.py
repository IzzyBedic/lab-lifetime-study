import pandas as pd

def add_lifespan_column(profile_path, data_path, output_path):
    """
    Merges lifespan (computed from birth and death dates) into the target dataset by subject_id.

    Parameters:
    - profile_path: str, path to dog_profile.csv
    - data_path: str, path to the target dataset (e.g., study_endpoints.csv)
    - output_path: str, path to save the new dataset with the lifespan column
    """

    # Load dog profile
    df_profile = pd.read_csv(profile_path)
    df_profile["birth_date"] = pd.to_datetime(df_profile["birth_date"], errors="coerce")
    df_profile["death_date"] = pd.to_datetime(df_profile["death_date"], errors="coerce")

    # Calculate lifespan in years
    df_profile["lifespan"] = (df_profile["death_date"] - df_profile["birth_date"]).dt.days / 365.25

    # Load the target dataset
    df_data = pd.read_csv(data_path)

    # Merge lifespan into the target dataset
    df_merged = df_data.merge(df_profile[["subject_id", "lifespan"]], on="subject_id", how="left")

    # Save to CSV
    df_merged.to_csv(output_path, index=False)
    print(f"✅ File saved as: {output_path}")


# Example usage
add_lifespan_column(
    profile_path="dog_profile.csv",
    data_path="poison_exposure.csv",  # change the target excell file you want to add lifespan column to it
    output_path="poison_exposure_with_lifespan.csv" # change the target excell file name after adding column
)
