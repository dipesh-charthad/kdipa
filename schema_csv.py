import psycopg2
from psycopg2 import sql
import pandas as pd
import os
import datetime
from dotenv import load_dotenv

load_dotenv()

def check_postgres_connection(host, database, user, password, port=5432):
    """
    Check connection to PostgreSQL database on Azure
    
    Args:
        host (str): Azure PostgreSQL server hostname
        database (str): Database name
        user (str): Username for database connection
        password (str): Password for database connection
        port (int, optional): Database port. Defaults to 5432.
    
    Returns:
        connection object if successful, None otherwise
    """
    try:
        # Establish connection
        connection = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password,
            port=port,
            # Optional SSL parameters for Azure
            sslmode='require'  # Azure typically requires SSL
        )
        
        # Create a cursor to execute a test query
        cursor = connection.cursor()
        cursor.execute("SELECT version();")
        db_version = cursor.fetchone()
        
        print("="*80)
        print(f"✅ SUCCESSFULLY CONNECTED TO DATABASE!")
        print(f"Database Version: {db_version[0]}")
        print("="*80)
        
        # Close cursor but return the connection for further use
        cursor.close()
        return connection
    
    except Exception as e:
        print("="*80)
        print(f"❌ ERROR CONNECTING TO POSTGRESQL DATABASE: {e}")
        print("="*80)
        return None

def list_tables(connection):
    """
    List all tables in the database and their row counts
    
    Args:
        connection: Active database connection
    
    Returns:
        list of tuples with (schema, table) information
    """
    try:
        # Create cursor
        cursor = connection.cursor()
        
        # Query to get all user tables (excluding system tables)
        cursor.execute("""
            SELECT 
                table_schema, 
                table_name 
            FROM information_schema.tables 
            WHERE table_type = 'BASE TABLE' 
            AND table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY table_schema, table_name
        """)
        
        # Fetch all tables
        tables = cursor.fetchall()
        
        # Check if any tables were found
        if not tables:
            print("\nNo tables found in the database.")
            cursor.close()
            return []
        
        # Group tables by schema
        schemas = {}
        for schema, table in tables:
            if schema not in schemas:
                schemas[schema] = []
            schemas[schema].append(table)
        
        # Print tables grouped by schema
        print(f"\n📊 FOUND {len(tables)} TABLES IN THE DATABASE:")
        print("="*80)
        
        for schema, tables_list in schemas.items():
            print(f"\nSchema: {schema}")
            for i, table in enumerate(tables_list, 1):
                # Get row count for each table
                try:
                    cursor.execute(
                        sql.SQL("SELECT COUNT(*) FROM {}.{}").format(
                            sql.Identifier(schema),
                            sql.Identifier(table)
                        )
                    )
                    row_count = cursor.fetchone()[0]
                    print(f"  {i}. {table} (Rows: {row_count})")
                except Exception as e:
                    print(f"  {i}. {table} (Could not get row count: {str(e)})")
        
        # Close cursor
        cursor.close()
        return tables
    
    except Exception as e:
        print(f"Error listing tables: {e}")
        return []

def export_tables_to_csv(connection, tables, output_dir=None, max_rows=None):
    """
    Export all tables to CSV files
    
    Args:
        connection: Active database connection
        tables: List of (schema, table) tuples
        output_dir: Directory to save CSV files (default: current datetime folder)
        max_rows: Maximum rows to export per table (None = all rows)
    
    Returns:
        List of paths to created CSV files
    """
    # Create output directory if none provided
    if not output_dir:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"postgres_export_{timestamp}"
    
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\nCreated output directory: {output_dir}")
    else:
        print(f"\nUsing existing output directory: {output_dir}")
    
    csv_files = []
    
    try:
        # Create cursor
        cursor = connection.cursor()
        
        # Loop through each table and export data
        for schema, table in tables:
            try:
                print(f"\nExporting {schema}.{table} to CSV...")
                
                # Create filename
                filename = f"{schema}_{table}.csv"
                filepath = os.path.join(output_dir, filename)
                
                # Prepare SQL query - with or without LIMIT
                if max_rows:
                    query = sql.SQL("SELECT * FROM {}.{} LIMIT %s").format(
                        sql.Identifier(schema),
                        sql.Identifier(table)
                    )
                    cursor.execute(query, [max_rows])
                else:
                    query = sql.SQL("SELECT * FROM {}.{}").format(
                        sql.Identifier(schema),
                        sql.Identifier(table)
                    )
                    cursor.execute(query)
                
                # Get column names
                col_names = [desc[0] for desc in cursor.description]
                
                # Fetch all rows
                rows = cursor.fetchall()
                
                if rows:
                    # Create DataFrame and export to CSV
                    df = pd.DataFrame(rows, columns=col_names)
                    df.to_csv(filepath, index=False)
                    csv_files.append(filepath)
                    
                    row_count = len(df)
                    col_count = len(df.columns)
                    size_kb = os.path.getsize(filepath) / 1024
                    
                    print(f"✅ Exported {row_count} rows, {col_count} columns to: {filepath} ({size_kb:.2f} KB)")
                else:
                    print(f"⚠️ No data to export for {schema}.{table}")
            
            except Exception as e:
                print(f"❌ Error exporting {schema}.{table}: {str(e)}")
        
        # Close cursor
        cursor.close()
        
        # Final summary
        if csv_files:
            print(f"\n{'='*80}")
            print(f"📂 SUCCESSFULLY EXPORTED {len(csv_files)} OF {len(tables)} TABLES TO CSV")
            print(f"📂 Output directory: {os.path.abspath(output_dir)}")
            print(f"{'='*80}")
        else:
            print(f"\n{'='*80}")
            print(f"❌ FAILED TO EXPORT ANY TABLES TO CSV")
            print(f"{'='*80}")
        
        return csv_files
    
    except Exception as e:
        print(f"Error in export_tables_to_csv: {e}")
        return csv_files

def main():
    load_dotenv()
    # Replace with your actual Azure PostgreSQL connection details
    HOST = os.getenv("HOST")
    DATABASE = os.getenv("DATABASE")
    USER = os.getenv("USER")
    PASSWORD = os.getenv("PASSWORD")
    PORT = os.getenv("PORT")
    
    # Set to None to export all rows, or a number to limit rows per table
    MAX_ROWS = None  # Set to None for all rows, or a number like 1000 to limit
    
    # Optional custom output directory (comment out to use default timestamp-based directory)
    # OUTPUT_DIR = "my_postgres_exports"
    
    # 1. Check connection and get active connection
    connection = check_postgres_connection(HOST, DATABASE, USER, PASSWORD, PORT)
    
    if connection:
        try:
            # 2. List tables and get table information
            tables = list_tables(connection)
            
            if tables:
                # 3. Export all tables to CSV files
                export_tables_to_csv(connection, tables, max_rows=MAX_ROWS)
        
        except Exception as e:
            print(f"An error occurred in main process: {e}")
        
        finally:
            # Always close the connection
            connection.close()
            print("\n"+"="*80)
            print("Database connection closed.")
            print("="*80)

if __name__ == "__main__":
    main()