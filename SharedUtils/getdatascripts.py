# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 15:57:46 2025

@author: Georg
"""
import subprocess
import os
import sqlite3
import pandas as pd


# Dump SQL using mysqldump
"""
    "c:\Program Files\MySQL\MySQL Server 9.3\bin\mysqldump.exe" 
    texas_schools > dump.sql
"""
# Convert to db file using sqlite3
"""
sqlite3 my_new_db.db
sqlite> .read backup.sql
sqlite> .quit
"""
#sqlite3mysql 
# Now I can use sqlite3 to read files from database

#Find the SQL install directory to run mysql commands
parent_directory = os.getcwd()
sql_dir = os.getenv("MYSQL_DIR")
mysqldump = os.path.join(sql_dir, "bin", "mysqldump.exe")
dm_dir = os.path.join(parent_directory, "DM")

def ExportCSV(dataframe, table_name, index_label):
    file_path = os.path.join(dm_dir, table_name + '.csv')
    dataframe.to_csv(file_path, index=True, index_label=index_label)
    return file_path

def DumpMYSQLDB(db_schema, sql_db_file):
    cmd_args = [mysqldump, db_schema, '>', sql_db_file]

    subprocess.Popen(cmd_args, shell=True)
    
def ConvertToSQLITEFile(db_schema, db_file):
    script_file = os.path.join(parent_directory, "Scripts", "mysql2sqlite")
    args = [script_file,
            '--sqlite-file',
            db_file,
            '--mysql-database',
            db_schema,
            '--mysql-user',
            'root',
            '--mysql-password',
            'st!xei@']

    try:
        proc = subprocess.Popen(args, shell=True)
        proc.wait()

    except Exception as e:
        print(f"An error occurred during the command run: {e}")
        
def ConverFromSQLITEFile(db_schema, db_file, table):
    script_file = os.path.join(parent_directory, "Scripts", "sqlite3mysql")
    args = [script_file,
            '--sqlite-file',
            db_file,
            '--sqlite-tables', 
            table,
            '--mysql-database',
            db_schema,
            '--mysql-user',
            'root',
            '--mysql-password',
            'st!xei@', 
            '--mysql-truncate-tables']

    try:
        process = subprocess.Popen(args, shell=True)
        process.wait()

    except Exception as e:
        print(f"An error occurred during the command run: {e}")


def GetDataFromDB(schema, dbfile, tables=[], refetch=True, allTables=True):
    """
    Gets a dictionary of dataframes from the database file 

    Parameters
    ----------
    schema : string
        database schema to fetch from.
    dbfile : string
        file name for db file to get data from.
    tables : list of strings, optional
        Only needed if allTables=False
        tables to fetch the dataframes for (each table -> dataframe).
    refetch : bool, optional
        If the script should refetch from the mysql server database. 
        The default is True.
    allTables : bool, optional
        Fetch all tables from schema. 
        The default is True.

    Returns
    -------
    dict Dataframes (keys = table names, values = dataframe for each table).

    """
    
    if(refetch):
        try:
            os.remove(dbfile)
            print(f"Deleted old db file '{dbfile}'.")
        except FileNotFoundError:
            print(f"'{dbfile}' does not exist, will create new one.")
            pass
        except Exception as e:
            print(f"An error occurred deleting old db file: {e}")
            pass
        ConvertToSQLITEFile(schema, dbfile)
    
    try:
        con = sqlite3.connect(dbfile)
        sql_tables = ""
        if(allTables):
            tables = [] # clear any tables they sent
            cur = con.cursor()
            sql_tables = "SELECT name from sqlite_master WHERE type='table';"
            res = cur.execute(sql_tables)
            tables_tmp = res.fetchall()
            for tmp in tables_tmp:
                str_table = str(tmp[0])
                tables.append(str_table)
            
        d_dataframes = {}
    
        for table in tables:
            sql = "SELECT * from " + table
            df = pd.read_sql(sql, con)
            
            d_dataframes[table] = df
        
        con.close()
    except sqlite3.OperationalError as e:
        print(f"An error occurred connecting to database: {e}")
    
    return d_dataframes


def CreateCleanedTable(cleaned_df, table, dbfile, index_label, schema):
    #sqlite> .mode csv
    # sqlite> .import c:/sqlite/city_no_header.csv cities
    
    con = sqlite3.connect(dbfile)
    cleaned_df.to_sql(table, con, if_exists='replace', 
                      index=True, index_label= index_label)
    con.close()
    
    ConverFromSQLITEFile(schema, dbfile, table)
    return None

    """
    Parameters
    ----------
    cleaned_df : TYPE
        DESCRIPTION.
    tables : TYPE
        DESCRIPTION.
    new_db_file : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """


#d_dataframes = GetDataFromDB("texas_schools", 'texas_schools.db')