import psycopg2 as pc
import pandas as pd 

class DBConnection:
    def __init__(self):
        self.conn = None

    def connect(self):
        try:
            host="watti.co9wquvnfbh3.ap-northeast-2.rds.amazonaws.com"
            dbname="EReport"
            user="postgres"
            password="start0212!"
            port=5432

            self.conn=pc.connect(host=host, dbname=dbname, user=user, password=password, port=port)

            return self.conn
        except pc.Error as e:
            # Handle connection error
            print(f"Database connection error: {e}")

    def execute_query(self, sql, params=None, dtype=None):
        try:
            cursor = self.conn.cursor()
            
            result = pd.read_sql_query(sql, self.conn, params=params, dtype=dtype)
            cursor.close()
            return result
        except pc.Error as e:
            # Handle query execution error
            print(f"Query execution error: {e}")
             

    def close_connection(self):
        if self.conn:
            self.conn.close()