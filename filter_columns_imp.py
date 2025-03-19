import pandas as pd

def filter_columns(df, columns_to_keep):
    """
    Filter DataFrame to keep only the specified columns.
    If a column in columns_to_keep doesn't exist in the DataFrame, it will be ignored.
    
    Args:
        df (pandas.DataFrame): The DataFrame to filter
        columns_to_keep (list): List of column names to keep
    
    Returns:
        pandas.DataFrame: DataFrame with only the specified columns
    """
    # Find which columns from the list actually exist in the DataFrame
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    
    # Return DataFrame with only these columns
    return df[existing_columns]

# Your list of columns to keep
columns_to_keep = [
    "estimation", "appType", "appState", "previousState", "appPhase", 
    "details", "trackId", "decisionDate", "processState", "type", "expiration", 
    "name", "proposedName", "licenseType", "details_company", "activityId", 
    "type", "sectorId", "formType", "formState", "details_form", "sectorId", 
    "activityId", "isDisabled", "creation", "name", "arabicName", 
    "isDisabled", "isArchived", "username", "type_user", "status"
]

# Example usage
# Assume you have a DataFrame called 'df' loaded from your data source
df = pd.read_csv(r"C:\Users\Dipesh.charthad\OneDrive - iLink Systems Inc\Desktop\Desktop\KDIPA\Code\excel_files\joined_tables.csv")  
# # or pd.read_excel(), etc.

# Filter the DataFrame
filtered_df = filter_columns(df, columns_to_keep)

# Save the filtered DataFrame to a new file
filtered_df.to_csv('details_form.csv', index=False)  # or to_excel(), etc.