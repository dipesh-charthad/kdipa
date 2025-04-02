import psycopg2
from psycopg2 import sql
import pandas as pd
import os
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

def fetch_table_data(connection, tables, max_rows=100):
    """
    Fetch and display data from all tables
    
    Args:
        connection: Active database connection
        tables: List of (schema, table) tuples
        max_rows: Maximum rows to fetch per table
    """
    try:
        # Create cursor
        cursor = connection.cursor()
        
        # Loop through each table and fetch data
        for schema, table in tables:
            print(f"\n{'='*80}")
            print(f"📋 TABLE: {schema}.{table}")
            print(f"{'='*80}")
            
            try:
                # Get column information
                cursor.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position
                """, (schema, table))
                
                columns = cursor.fetchall()
                print("Columns:")
                for col_name, col_type in columns:
                    print(f"  {col_name} ({col_type})")
                
                # Count rows in table
                cursor.execute(
                    sql.SQL("SELECT COUNT(*) FROM {}.{}").format(
                        sql.Identifier(schema),
                        sql.Identifier(table)
                    )
                )
                total_rows = cursor.fetchone()[0]
                
                # Fetch data with row limit
                cursor.execute(
                    sql.SQL("SELECT * FROM {}.{} LIMIT %s").format(
                        sql.Identifier(schema),
                        sql.Identifier(table)
                    ),
                    [max_rows]
                )
                
                rows = cursor.fetchall()
                
                if rows:
                    # Get column names
                    col_names = [desc[0] for desc in cursor.description]
                    
                    # Use pandas to display the data nicely
                    df = pd.DataFrame(rows, columns=col_names)
                    print(f"\nShowing {len(rows)} of {total_rows} rows:")
                    
                    # Set display options for better view
                    pd.set_option('display.max_columns', 10)
                    pd.set_option('display.width', 1000)
                    pd.set_option('display.max_colwidth', 50)
                    
                    print(df)
                    
                    # Reset display options
                    pd.reset_option('display.max_columns')
                    pd.reset_option('display.width')
                    pd.reset_option('display.max_colwidth')
                    
                    # If we have more columns than can be displayed, show a message
                    if len(df.columns) > 10:
                        print(f"Note: Not all columns may be visible. DataFrame has {len(df.columns)} columns total.")
                else:
                    print("\nNo data in this table.")
            
            except Exception as e:
                print(f"Error fetching data from {schema}.{table}: {str(e)}")
        
        # Close cursor
        cursor.close()
    
    except Exception as e:
        print(f"Error in fetch_table_data: {e}")

def main():
    load_dotenv()
    # Replace with your actual Azure PostgreSQL connection details
    HOST = os.getenv("HOST")
    DATABASE = os.getenv("DATABASE")
    USER = os.getenv("USER")
    PASSWORD = os.getenv("PASSWORD")
    PORT = os.getenv("PORT")
    
    # Maximum rows to fetch per table
    MAX_ROWS = 100
    
    # 1. Check connection and get active connection
    connection = check_postgres_connection(HOST, DATABASE, USER, PASSWORD, PORT)
    
    if connection:
        try:
            # 2. List tables and get table information
            tables = list_tables(connection)
            
            if tables:
                # 3. Fetch and display data from all tables
                fetch_table_data(connection, tables, MAX_ROWS)
        
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
