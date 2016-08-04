#!/usr/bin/env python3

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, drop_database, create_database
from dbutils import *
import pandas as pd
import datetime
import argparse
import sys
import os

deg = 180.0 / np.pi


def parse():
    parser = argparse.ArgumentParser(description="Make database for time series data.")
    parser.add_argument('-d', dest='directory', type=str, default=None, action='store',
        help='Parent directory containing time series data. Default: None.')
    parser.add_argument('-f', type=str, action='store', default=None, dest='filelist',
        help='Input file list containing path to data files. Default: None.')
    parser.add_argument('-v', '-V', action='store_true', dest='verbose',
        help='Verbose mode.')
    parser.add_argument('-o', type=str, action='store', default=None, dest='output',
        help='Output file path for database. Default: None.')
    return parser.parse_args()


def main(inputs):

    # Read list of files or build
    if inputs.directory is None and inputs.filelist is not None:
        filelist = np.loadtxt(inputs.filelist, dtype=bytes).astype(str)
    elif inputs.directory is not None and inputs.filelist is None:
        fname = buildFileList(inputs.directory, verbose=inputs.verbose)
        filelist = np.loadtxt(fname, dtype=bytes).astype(str)
    else:
        assert False, 'Must provide input directory or file list'

    # Initialize engine for SQL database
    dbname = inputs.output or '/net/jokull/bak/geonet/aux_data/raw_data.db'
    engine = create_engine('sqlite:///%s' % dbname)

    # If database doesn't exist, create it and initialize output SQL tables
    ref_meta = None
    if not os.path.isfile(dbname):
        # Create database
        create_database(engine.url)
        # Initialize data table
        cmd = (
            "CREATE TABLE tseries("
            "DATE TEXT, "
            "east FLOAT, "
            "north FLOAT, "
            "up FLOAT, "
            "sigma_east FLOAT, "
            "sigma_north FLOAT, "
            "sigma_up FLOAT, "
            "id TEXT);")
        pd.io.sql.execute(cmd, engine)
        # Initialize file list table
        pd.io.sql.execute("CREATE TABLE files(path TEXT);", engine)

    # If it does exist, read the list of files that already exist in the 
    # database and only keep the new ones
    else:
        if inputs.verbose: print('Constructing list of unique files')
        file_df = pd.read_sql_table('files', engine)
        ref_files = file_df['path'].values
        filelist = np.setxor1d(ref_files, filelist)

        # Also read reference metadata
        ref_meta = pd.read_sql_table('metadata', engine, 
            columns=['id', 'lon', 'lat', 'elev'])

    if inputs.verbose:
        print('Number of new files to add to database:', len(filelist))

    # Create standard insertion SQL command
    cmd = ("INSERT INTO tseries(DATE, east, north, up, "
           "sigma_east, sigma_north, sigma_up, id) "
           "VALUES(?, ?, ?, ?, ?, ?, ?, ?);")

    # Loop over new filenames 
    cnt = 0
    stations = []; longitude = []; latitude = []; elev = []
    for filepath in filelist:

        if cnt % 100 == 0 and inputs.verbose:
            sys.stdout.write(' - %d\r' % cnt)
            sys.stdout.flush()

        name = filepath.split('/')[-1]

        # Skip if not a final solution
        if '.tseries' not in name:
            continue
        if '.final.' not in name:
            continue

        # Parse the filename
        statname = name[:4].lower()

        # Load the data
        data = np.loadtxt(filepath)
        # Save the date
        year, month, day, hour = data[9:13].astype(int)
        date = datetime.datetime(year, month, day, hour)
        # Read the coordinates if they aren't stored already
        if statname not in stations:
            with open(filepath, 'r') as ifid:
                for line in ifid:
                    if 'STA X' in line:
                        statX = float(line.split()[5])
                    elif 'STA Y' in line:
                        statY = float(line.split()[5])
                    elif 'STA Z' in line:
                        statZ = float(line.split()[5])
                    elif 'SRGD' in line:
                        break
            lat,lon,h = xyz2llh(np.array([statX, statY, statZ]))
            stations.append(statname)
            longitude.append(lon*deg)
            latitude.append(lat*deg)
            elev.append(h)

        # Create tuple of new data
        data = (date, data[0], data[1], data[2], data[3], data[4], data[5], statname)

        # Add data to SQL database
        pd.io.sql.execute(cmd, engine, params=[data])
        # And the file path
        pd.io.sql.execute('INSERT INTO files(path) VALUES(?);', engine, params=[(filepath,)])
        cnt += 1

    print('')

    # Make unique metadata table
    meta_df = pd.DataFrame({'id': stations, 'lon': longitude, 'lat': latitude, 'elev': elev})
    if ref_meta is None:
        meta_df.to_sql('metadata', engine)
    else:
        meta_df = pd.concat([meta_df, ref_meta]).drop_duplicates(subset='id')
        meta_df.to_sql('metadata', engine, if_exists='replace')


if __name__ == '__main__':
    inputs = parse()
    main(inputs)

# end of file
