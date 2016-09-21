#-*- coding: utf-8 -*-

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, drop_database, create_database
import pandas as pd
import numpy as np
import sys


class Engine:
    """
    Utility class for initializing a database engine.
    """

    def __init__(self, dbname=None, dbtype=None, url=None):
        """
        Initialize appropriate engine and url.
        """
        # If url is provided, create database directly
        if url is not None:
            engine = create_engine(url)
            dbname = url.split('/')[-1]

        else:
            # sqlite creates a file on disk
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


    def initdb(self, new=False, ref_engine=None):
        """
        Initialize a time series table given a creation command string.
        """
        # If new database is requested, drop any existing database
        if new and database_exists(self.url):
            drop_database(self.url)
            create_database(self.url)
        # Initialize file list table
        if 'files' not in self.tables(asarray=True):
            pd.io.sql.execute("CREATE TABLE files(path TEXT);", self.engine)
        # If a reference engine is provided, copy over necessary bits
        if ref_engine is not None:
            meta = ref_engine.meta()
            meta.to_sql('metadata', self.engine, if_exists='replace')
            file_df = pd.read_sql_table('files', ref_engine.engine, columns=['path',])
            file_df.to_sql('files', self.engine, if_exists='replace')
            
        return


    def getUniqueFiles(self, newlist=None):
        """
        Return list of filenames present in the database. If newlist is not None,
        then find the set exclusive of the two arrays.
        """
        try:
            file_df = pd.read_sql_table('files', self.engine)
            ref_files = file_df['path'].values
            if newlist is None:
                return ref_files
            else:
                return np.setxor1d(ref_files, newlist)
        except ValueError:
            if type(newlist) in [list, tuple, np.ndarray]:
                return newlist
            else:
                return []

 
    def meta(self):
        """
        Return metadata table.
        """
        try:
            metadata = pd.read_sql_table('metadata', self.engine, 
                columns=['id', 'lon', 'lat', 'elev'])
        except ValueError:
            metadata = pd.DataFrame({'id': [], 'lat': [], 'lon': [], 'elev': []})
        return metadata


    def updateMeta(self, statlist):
        """
        Update metadata with list subset.
        """
        # Compare ID list
        meta_ref = self.meta()
        names = meta_ref['id'].values
        ind = np.in1d(names, statlist)
        num_good_stat = len(ind.nonzero()[0])

        # If changes need to be made
        if num_good_stat != len(names):

            # Subset the metadata            
            meta_sub = meta_ref.loc[ind,:].reset_index()

            # Update the filelist
            file_df = pd.read_sql_table('files', self.engine, columns=['path',])
            keep_file = []
            for index, path in enumerate(file_df['path']):
                filename = path.split('/')[-1].lower()
                for stat_id in statlist:
                    if stat_id in filename:
                        keep_file.append(index)

            # Write to database
            meta_sub.to_sql('metadata', self.engine, if_exists='replace')
            file_df = file_df.loc[keep_file].reset_index()
            file_df.to_sql('files', self.engine, if_exists='replace')
            
            return meta_sub, True
        else:
            return None, False


    def dates(self):
        """
        Return an array of dates.
        """
        components = self.components()
        df = pd.read_sql_query("SELECT DATE FROM %s;" % components[0], self.engine)
        return pd.to_datetime(df['DATE'], format='%Y-%m-%d %H:%M:%S.%f').values


    def addFile(self, filepath):
        """
        Add file path to files table.
        """
        if isinstance(filepath, list):
            df = pd.DataFrame({'path': filepath})
            df.to_sql('files', self.engine, if_exists='replace')
        elif isinstance(filepath, str):
            pd.io.sql.execute('INSERT INTO files(path) VALUES(?);', 
                self.engine, params=[(filepath,)])
        return


    def tables(self, asarray=False):
        """
        Get list of tables stored in a SQL database.
        """
        cmd = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = pd.read_sql_query(cmd, self.engine)
        if asarray:
            tables = tables['name'].values
        return tables


    def components(self):
        """
        Determine the component types stored in the database using the 
        table list.
        """
        tables = self.tables()
        comps = []
        for name in tables['name'].values:
            if '_' not in name and name not in ['files', 'metadata']:
                comps.append(name)
        return sorted(comps)


# end of main
