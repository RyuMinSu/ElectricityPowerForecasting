import pandas as pd

import pymysql
import sqlalchemy
from sqlalchemy import create_engine

#csv > db
def createTb(dfPath, id, pw, host, dbName, tbName, exists):
    df = pd.read_csv(dfPath)
    dbConnPath = f"mysql + pymysql://{id}:{pw}@{host}/{dbName}"
    dbConn = create_engine(dbConnPath)
    conn = dbConn.connect()
    df.to_sql(name=tbName, con=dbConn, if_exists=exists, index=False)
    print(f"successful create {tbName}  in {dbName} DataBase")

#db > csv
def readTb(host, id, pw, dbName, rsql):
    conn = pymysql.connect(host=host, user=id, passwd=pw,\
        db=dbName, charset="utf8")
    cur = conn.cursor()
    df = pd.read_sql(rsql, con=conn)
    print(f"df shape: {df.shape}")
    return df