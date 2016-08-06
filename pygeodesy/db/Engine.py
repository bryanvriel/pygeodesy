#-*- coding: utf-8 -*-

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, drop_database, create_database
import pandas as pd
import numpy as np


class Engine:
    """
    Utility class for initializing a database engine.
    """

    def __init__(self, dbname, dbtype):
        """
        Initialize appropriate engine and url.
        """
        # sqlite creates a file on dirk
        if dbtype == 'sqlite':
            engine = create_engine('sqlite:///%s' % dbname)

        # MySQL needs a server running
        elif dbtype == 'mysql':
            import socket
            computer_name = socket.gethostname()
            engine = create_engine('mysql+pymysql://root:%s@localhost:3306/%s' % 
                (computer_name, dbname))

        self.dbname = dbname
        self.engine = engine
        self.url = engine.url

        return


    def initdb(self, cmd):
        """
        Initialize a time series table given a creation command string.
        """
        # Create the database
        create_database(self.url)
        # Create the table
        #pd.io.sql.execute(cmd, self.engine)
        # Initialize file list table
        pd.io.sql.execute("CREATE TABLE files(path TEXT);", self.engine)


    def getUniqueFiles(self, newlist=None):
        """
        Return list of filenames present in the database. If newlist is not None,
        then find the set exclusive of the two arrays.
        """
        file_df = pd.read_sql_table('files', self.engine)
        ref_files = file_df['path'].values
        if newlist is None:
            return ref_file
        else:
            return np.setxor1d(ref_files, newlist)

 
    def meta(self):
        """
        Return metadata table.
        """
        metadata = pd.read_sql_table('metadata', self.engine, 
            columns=['id', 'lon', 'lat', 'elev'])
        return metadata


    def addData(self, data_df):
        """
        Add data record(s) to time series table.
        """
        data_df.to_sql('tseries', self.engine, if_exists='append') 

        #if data.ndim == 2:
        #    for count in range(data.shape[0]):
        #        out = [date[count]] + list(data[count,:]) + [name]
        #        pd.io.sql.execute(cmd, self.engine, params=out)
        #else:
        #    out = [date] + list(data) + [name]
        #    pd.io.sql.execute(cmd, self.engine, params=out)

        return


    def addFile(self, filepath):
        """
        Add file path to files table.
        """
        pd.io.sql.execute('INSERT INTO files(path) VALUES(?);', 
            self.engine, params=[(filepath,)])
        return


# end of main
