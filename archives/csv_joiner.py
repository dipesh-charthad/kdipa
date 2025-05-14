import pandas as pd
import os
from pathlib import Path

def join_tables(folder_path):
    """
    Joins all database tables through common fields and outputs the result to a CSV file.
    
    Args:
        folder_path (str): Path to the folder containing CSV files for each table
    
    Returns:
        pd.DataFrame: The joined dataframe
    """
    folder_path = Path(folder_path)
    
    # Load all tables from CSV files
    print("Loading tables...")
    activity = pd.read_csv(r"C:\Users\Dipesh.charthad\OneDrive - iLink Systems Inc\Desktop\Desktop\KDIPA\Code\postgres_export_20250307_095845\public_activity_view.csv")
    application = pd.read_csv(r"C:\Users\Dipesh.charthad\OneDrive - iLink Systems Inc\Desktop\Desktop\KDIPA\Code\postgres_export_20250307_095845\public_application_view.csv")
    client = pd.read_csv(r"C:\Users\Dipesh.charthad\OneDrive - iLink Systems Inc\Desktop\Desktop\KDIPA\Code\postgres_export_20250307_095845\public_client_view.csv")
    company = pd.read_csv(r"C:\Users\Dipesh.charthad\OneDrive - iLink Systems Inc\Desktop\Desktop\KDIPA\Code\postgres_export_20250307_095845\public_company_view.csv")
    form_activity_sector = pd.read_csv(r"C:\Users\Dipesh.charthad\OneDrive - iLink Systems Inc\Desktop\Desktop\KDIPA\Code\postgres_export_20250307_095845\public_form_activity_sector_view.csv")
    form = pd.read_csv(r"C:\Users\Dipesh.charthad\OneDrive - iLink Systems Inc\Desktop\Desktop\KDIPA\Code\postgres_export_20250307_095845\public_form_view.csv")
    sector_activity = pd.read_csv(r"C:\Users\Dipesh.charthad\OneDrive - iLink Systems Inc\Desktop\Desktop\KDIPA\Code\postgres_export_20250307_095845\public_sector_activity_view.csv")
    sector = pd.read_csv(r"C:\Users\Dipesh.charthad\OneDrive - iLink Systems Inc\Desktop\Desktop\KDIPA\Code\postgres_export_20250307_095845\public_sector_view.csv")
    user = pd.read_csv(r"C:\Users\Dipesh.charthad\OneDrive - iLink Systems Inc\Desktop\Desktop\KDIPA\Code\postgres_export_20250307_095845\public_user_view.csv")
    
    print("Joining tables...")
    
    # Start with application as the base table
    result = application.copy()
    
    # Join with form
    result = result.merge(
        form,
        left_on="uuid",
        right_on="applicationUuid",
        how="left",
        suffixes=("", "_form")
    )
    
    # Join with company
    result = result.merge(
        company,
        left_on="companyUuid",
        right_on="uuid",
        how="left",
        suffixes=("", "_company")
    )
    
    # Join with client for company
    result = result.merge(
        client,
        left_on="clientUuid",
        right_on="uuid",
        how="left",
        suffixes=("", "_client_company")
    )
    
    # Join with user 
    result = result.merge(
        user,
        left_on="userUuid",
        right_on="uuid",
        how="left",
        suffixes=("", "_user")
    )
    
    # Join with client for user
    result = result.merge(
        client.rename(columns={"uuid": "uuid_client_user", "modification": "modification_client_user", "type": "type_client_user"}),
        left_on="clientUuid_user",
        right_on="uuid_client_user",
        how="left"
    )
    
    # Join form with form_activity_sector
    result = result.merge(
        form_activity_sector,
        left_on="uuid_form",
        right_on="formUUID",
        how="left",
        suffixes=("", "_form_act_sect")
    )
    
    # Join with sector_activity
    result = result.merge(
        sector_activity,
        left_on=["sectorId", "activityId"],
        right_on=["sectorId", "activityId"],
        how="left",
        suffixes=("", "_sect_act")
    )
    
    # Join with activity
    result = result.merge(
        activity,
        left_on="activityId",
        right_on="id",
        how="left",
        suffixes=("", "_activity")
    )
    
    # Join with sector
    result = result.merge(
        sector,
        left_on="sectorId",
        right_on="id",
        how="left",
        suffixes=("", "_sector")
    )
    
    # Output to CSV
    output_path = r"C:\Users\Dipesh.charthad\OneDrive - iLink Systems Inc\Desktop\Desktop\KDIPA\Code\postgres_export_20250307_095845\joined_tables.csv"
    result.to_csv(output_path, index=False)
    print(f"Joined table saved to {output_path}")
    
    return result

def main():
    print("Database Table Joiner")
    print("=====================")
    folder_path = input("Enter the path to the folder containing CSV files: ")
    
    if not os.path.exists(folder_path):
        print(f"Error: The path '{folder_path}' does not exist.")
        return
    
    try:
        joined_data = join_tables(folder_path)
        print(f"Successfully joined {len(joined_data)} rows of data")
    except Exception as e:
        print(f"Error joining tables: {str(e)}")

if __name__ == "__main__":
    main()
