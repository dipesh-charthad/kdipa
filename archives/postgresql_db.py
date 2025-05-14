## For Checking Connection
# import os
# import psycopg2
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# # Get database credentials from environment variables
# HOST = os.getenv("HOST")
# DATABASE = os.getenv("DATABASE")
# USER = os.getenv("USER")
# PASSWORD = os.getenv("PASSWORD")
# PORT = os.getenv("PORT")

# try:
#     # Connect to the database
#     conn = psycopg2.connect(
#         dbname=DATABASE,
#         user=USER,
#         password=PASSWORD,
#         host=HOST,
#         port=PORT,
#         sslmode="require"
#     )
#     cursor = conn.cursor()
    
#     # Step 1: Print the current database
#     cursor.execute("SELECT current_database();")
#     print("Connected to database:", cursor.fetchone()[0])

#     cursor.execute("SELECT pg_size_pretty(pg_database_size(current_database())) AS database_size;")
#     print("Database Size:", cursor.fetchone()[0])
    
#     # Step 2: List all tables in the database
#     cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
#     tables = cursor.fetchall()
#     print("\nTables in the database:", [table[0] for table in tables])

#     # Check if 'documents' table exists
#     table_name = "documents"  # Change this if needed
#     if (table_name,) not in tables:
#         print(f"\nError: Table '{table_name}' does not exist!")
#     else:
#         # Step 3: Fetch all records from the 'documents' table
#         cursor.execute(f"SELECT * FROM {table_name} LIMIT 10;")  # Adjust LIMIT as needed
#         rows = cursor.fetchall()
        
#         # Print records
#         if rows:
#             print("\nDocuments in the table:")
#             for row in rows:
#                 print(row)
#         else:
#             print("\nNo records found in the 'documents' table.")

#     # Close connection
#     cursor.close()
#     conn.close()

# except Exception as e:
#     print("\nError:", e)


## For Create Table
import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get database credentials from environment variables
HOST = os.getenv("HOST")
DATABASE = os.getenv("DATABASE")
USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")
PORT = os.getenv("PORT")

try:
    # Connect to the database
    conn = psycopg2.connect(
        dbname=DATABASE,
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        sslmode="require"
    )
    conn.autocommit = True  # Enable autocommit for table creation
    cursor = conn.cursor()
    
    # Step 1: Print the current database
    cursor.execute("SELECT current_database();")
    print("Connected to database:", cursor.fetchone()[0])

    cursor.execute("SELECT pg_size_pretty(pg_database_size(current_database())) AS database_size;")
    print("Database Size:", cursor.fetchone()[0])
    
    # Step 2: List all tables in the database
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
    tables = cursor.fetchall()
    print("\nTables in the database:", [table[0] for table in tables])

    # Create a dummy table with 2 columns
    dummy_table_name = "dummy_table"
    try:
        print(f"\nCreating dummy table '{dummy_table_name}'...")
        # cursor.execute(f"""
        # CREATE TABLE IF NOT EXISTS {dummy_table_name} (
        #     id SERIAL PRIMARY KEY,
        #     name VARCHAR(100) NOT NULL
        # )
        # """)
        
        # # Insert some dummy data
        # cursor.execute(f"""
        # INSERT INTO {dummy_table_name} (name) VALUES 
        # ('Item 1'),
        # ('Item 2'),
        # ('Item 3'),
        # ('Item 4'),
        # ('Item 5')
        # ON CONFLICT DO NOTHING
        # """)
        
        # print(f"Dummy table '{dummy_table_name}' created successfully!")
        
        # Run SELECT command on the dummy table
        cursor.execute(f"SELECT * FROM {dummy_table_name};")
        dummy_rows = cursor.fetchall()
        
        print(f"\nData in the '{dummy_table_name}' table:")
        for row in dummy_rows:
            print(row)
            
    except Exception as e:
        print(f"Error creating dummy table: {e}")

    # Check if 'documents' table exists
    table_name = "documents"  # Change this if needed
    if (table_name,) not in tables:
        print(f"\nError: Table '{table_name}' does not exist!")
    else:
        # Step 3: Fetch all records from the 'documents' table
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 10;")  # Adjust LIMIT as needed
        rows = cursor.fetchall()
        
        # Print records
        if rows:
            print("\nDocuments in the table:")
            for row in rows:
                print(row)
        else:
            print("\nNo records found in the 'documents' table.")

    # Close connection
    cursor.close()
    conn.close()

except Exception as e:
    print("\nError:", e)


## For Delete Table
# import os
# import psycopg2
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# # Get database credentials from environment variables
# HOST = os.getenv("HOST")
# DATABASE = os.getenv("DATABASE")
# USER = os.getenv("USER")
# PASSWORD = os.getenv("PASSWORD")
# PORT = os.getenv("PORT")

# try:
#     # Connect to the database
#     conn = psycopg2.connect(
#         dbname=DATABASE,
#         user=USER,
#         password=PASSWORD,
#         host=HOST,
#         port=PORT,
#         sslmode="require"
#     )
#     conn.autocommit = True  # Enable autocommit for table deletion
#     cursor = conn.cursor()
    
#     # Table to delete
#     dummy_table_name = "dummy_table"
    
#     # Check if the table exists before attempting to delete
#     cursor.execute("""
#     SELECT EXISTS (
#         SELECT FROM information_schema.tables 
#         WHERE table_schema = 'public' 
#         AND table_name = %s
#     );
#     """, (dummy_table_name,))
    
#     table_exists = cursor.fetchone()[0]
    
#     if table_exists:
#         # Delete the table
#         print(f"Deleting table '{dummy_table_name}'...")
#         cursor.execute(f"DROP TABLE {dummy_table_name};")
#         print(f"Table '{dummy_table_name}' successfully deleted.")
#     else:
#         print(f"Table '{dummy_table_name}' does not exist.")
    
#     # Close connection
#     cursor.close()
#     conn.close()
#     print("Database connection closed.")

# except Exception as e:
#     print("\nError:", e)
