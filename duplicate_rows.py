import pandas as pd

def filter_csv(input_file, output_file):
    """
    Filter a CSV file to remove rows where formType is 'EMPLOYEE_INFO' and formState is 'INTERNAL'
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the filtered CSV file
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Print initial information
    print(f"Original CSV has {len(df)} rows.")
    
    # Filter out rows where formType is 'EMPLOYEE_INFO' and formState is 'INTERNAL'
    filtered_df = df[(df['formType'] != 'EMPLOYEE_INFO') | (df['formState'] != 'INTERNAL')]
    
    # Print how many rows were removed
    rows_removed = len(df) - len(filtered_df)
    print(f"Removed {rows_removed} rows where formType='EMPLOYEE_INFO' and formState='INTERNAL'.")
    
    # Save the filtered data to a new CSV file
    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered CSV saved to {output_file} with {len(filtered_df)} rows.")

# Example usage
if __name__ == "__main__":
    input_file = r"C:\Users\Dipesh.charthad\OneDrive - iLink Systems Inc\Desktop\Desktop\KDIPA\Code\details_form_filtered_20250319_123031.csv"
    output_file = r"C:\Users\Dipesh.charthad\OneDrive - iLink Systems Inc\Desktop\Desktop\KDIPA\Code\filtered_output_details_form.csv"
    filter_csv(input_file, output_file)