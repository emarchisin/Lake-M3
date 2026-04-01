import pandas as pd
from pathlib import Path
from tqdm import tqdm

def remove_old_parquet_data(root_folder: str, cutoff_date_str: str):
    """
    Traverses subfolders to find parquet files, removes rows where 'datetime' 
    is before the cutoff, and overwrites the files.
    """
    cutoff_date = pd.to_datetime(cutoff_date_str)
    base_path = Path(root_folder)
    
    if not base_path.is_dir():
        print(f"Error: The directory '{root_folder}' does not exist.")
        return

    # Convert generator to a list so tqdm knows the total count for the ETA
    parquet_files = list(base_path.rglob('*.parquet'))
    
    if not parquet_files:
        print(f"No parquet files found in {root_folder}")
        return

    # Wrap the list in tqdm() to generate the progress bar
    for file_path in tqdm(parquet_files, desc="Trimming Data", unit="file"):
        try:
            # Read the parquet file
            df = pd.read_parquet(file_path)
            original_size = len(df)
            
            # Filter the dataframe
            df_filtered = df[df['datetime'] >= cutoff_date]
            filtered_size = len(df_filtered)
            
            # Skip saving if no rows were removed
            if original_size == filtered_size:
                continue

            # Overwrite the original file
            df_filtered.to_parquet(file_path, index=False, compression='zstd')
            
        except KeyError:
            tqdm.write(f"Error: 'datetime' column not found in {file_path.name}")
        except Exception as e:
            tqdm.write(f"Unexpected error processing {file_path.name}: {e}")

# Main execution block -----------------------
if __name__ == "__main__":
    TARGET_FOLDER = "Project/gradients/export"
    CUTOFF_DATE = "2018-01-01" 
    
    remove_old_parquet_data(TARGET_FOLDER, CUTOFF_DATE)