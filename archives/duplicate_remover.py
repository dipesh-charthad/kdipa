import pandas as pd
import os
from pathlib import Path

def remove_duplicates(file_path, subset=None, keep='first', output_path=None):
    """
    Removes duplicate entries from a CSV file.
    
    Args:
        file_path (str): Path to the input CSV file
        subset (list, optional): List of column names to consider for identifying duplicates.
                                If None, all columns are used.
        keep (str, optional): Which duplicates to keep.
                             'first' - Keep first occurrence (default)
                             'last' - Keep last occurrence
                             False - Drop all duplicates
        output_path (str, optional): Path where to save the deduplicated CSV file.
                                   If None, the output is saved as "[original_filename]_no_duplicates.csv"
    
    Returns:
        tuple: (df, duplicate_count) - DataFrame after removing duplicates and the count of removed duplicates
    """
    # Convert to Path object
    file_path = Path(r"C:\Users\Dipesh.charthad\OneDrive - iLink Systems Inc\Desktop\Desktop\KDIPA\Code\postgres_export_20250307_095845\joined_tables.csv")
    
    # Read the CSV file
    print(f"Reading file: {file_path}")
    df = pd.read_csv(file_path)
    
    # Get initial row count
    initial_count = len(df)
    print(f"Initial row count: {initial_count}")
    
    # Remove duplicates
    if subset:
        print(f"Removing duplicates based on columns: {subset}")
        df_no_duplicates = df.drop_duplicates(subset=subset, keep=keep)
    else:
        print("Removing duplicates based on all columns")
        df_no_duplicates = df.drop_duplicates(keep=keep)
    
    # Get final row count
    final_count = len(df_no_duplicates)
    duplicate_count = initial_count - final_count
    
    # Print statistics
    print(f"Duplicates removed: {duplicate_count} ({duplicate_count/initial_count*100:.2f}%)")
    print(f"Final row count: {final_count}")
    
    # Save to CSV
    if output_path is None:
        output_path = file_path.parent / f"{file_path.stem}_no_duplicates.csv"
    
    df_no_duplicates.to_csv(output_path, index=False)
    print(f"Deduplicated data saved to: {output_path}")
    
    return df_no_duplicates, duplicate_count

def main():
    print("CSV Duplicate Remover")
    print("====================")
    
    # Get file path
    file_path = input("Enter the path to the CSV file: ")
    
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return
    
    # Ask if user wants to specify columns
    specify_columns = input("Do you want to specify columns for deduplication? (y/n): ").strip().lower()
    
    subset = None
    if specify_columns == 'y':
        columns_input = input("Enter column names separated by commas: ")
        subset = [col.strip() for col in columns_input.split(',')]
    
    # Ask which duplicates to keep
    keep_option = input("Which duplicates to keep? (first/last/none): ").strip().lower()
    if keep_option == 'first':
        keep = 'first'
    elif keep_option == 'last':
        keep = 'last'
    elif keep_option == 'none':
        keep = False
    else:
        print("Invalid option. Defaulting to 'first'.")
        keep = 'first'
    
    # Ask for output path
    output_path = input("Enter output file path (leave blank for default): ").strip()
    if not output_path:
        output_path = None
    
    try:
        remove_duplicates(file_path, subset, keep, output_path)
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
