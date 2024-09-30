# sqlite_util.py

import sqlite3
import pandas as pd
import os

CREATE_TABLE_STATEMENT = """ 
CREATE TABLE IF NOT EXISTS semantic_embeddings(
chunk_id INTEGER PRIMARY KEY,
chunk_text TEXT,
vector_embeddings TEXT,
document_id INTEGER,
document_title TEXT
)
"""
output_dir = "./llm-projects/hybrid_search"
# Create database directory path if it doesn't exist
os.makedirs(os.path.join(output_dir, "db"), exist_ok=True)
db = os.path.join(output_dir, "db", "semantic_search.db")


'''
* Function to create database
'''

def connect_to_db() -> sqlite3.Connection:
    #create a database
    connection = sqlite3.connect(db)
    return connection

'''
* Functio to create table.
'''
def create_table(connection : sqlite3.Connection):
    try:
        cursor = connection.cursor()
        cursor.execute(CREATE_TABLE_STATEMENT)
    except:
        raise ConnectionError("Error in creating cursor")
    finally:
        connection.close()

'''
* Function to write to sql from dataframe
'''
def write_to_db_from_dataframe(conn : sqlite3.Connection, 
                                df : pd.DataFrame):
    try:
        df.to_sql('semantic_embeddings', conn,
                   if_exists='replace', index=False)
    except:
        raise ConnectionError("Error in connection")
        raise Exception()
    finally:
        conn.close()

'''
* Function to retrieve data.
'''
def load_data_to_dataframe(conn : sqlite3.Connection, 
                          chunk_ids : list) -> pd.DataFrame:
    
    placeholders = ','.join('?' for _ in chunk_ids)
    query = f"SELECT * FROM semantic_embeddings WHERE chunk_id IN ({placeholders})"   
    
    df = pd.read_sql(query, conn, params = chunk_ids)
    
    return df

def test_classes():
    conn = connect_to_db()
    create_table(conn)

if __name__ == "__main__":
    test_classes()