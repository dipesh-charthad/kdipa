import pandas as pd
import os
from datetime import datetime

def filter_csv(input_file, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get the base filename without extension
    base_filename = os.path.basename(input_file).split('.')[0]
    
    # Create output filename
    output_file = os.path.join(output_dir, f"{base_filename}_filtered_{timestamp}.csv")
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Filter rows where appstate is either "accepted" or "rejected"
    filtered_df = df[df['appState'].isin(['ACCEPTED', 'REJECTED'])]
    
    # Save the filtered data to a new CSV file
    filtered_df.to_csv(output_file, index=False)
    
    print(f"Filtering complete. Original rows: {len(df)}, Filtered rows: {len(filtered_df)}")
    print(f"Filtered data saved to: {output_file}")
    
    return output_file

if __name__ == "__main__":
    # Example usage
    input_file = r"C:\Users\Dipesh.charthad\OneDrive - iLink Systems Inc\Desktop\Desktop\KDIPA\Code\archives\code\details_form.csv"  # Change this to your input file path
    output_dir = r"C:\Users\Dipesh.charthad\OneDrive - iLink Systems Inc\Desktop\Desktop\KDIPA\Code"  # Change this to your desired output directory
    
    filtered_file = filter_csv(input_file, output_dir)
