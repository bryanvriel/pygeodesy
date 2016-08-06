#-*- coding: utf-8

from __future__ import print_function
from .Engine import Engine
from .dbutils import *
import pandas as pd
import datetime
import sys
import os

import pygeodesy.instrument as instrument
import pygeodesy.network as network

# Define the default options
defaults = {
    'columns': "{'east': 0, 'north': 1, 'up': 2, 'sigma_east': 3, "
        "'sigma_north': 4, 'sigma_up': 5}",
    'columns': None,
    'format': None,
    'directory': None,
    'type': 'gps',
    'dbname': 'data.db',
    'dbtype': 'sqlite',
    'filelist': None,
    'metafile': None,
    'metafmt': None,
}


def makedb(optdict):

    # Update the options
    opts = defaults.copy()
    opts.update(optdict)

    # Read list of files or build
    if opts['directory'] is None and opts['filelist'] is not None:
        filelist = np.loadtxt(opts.filelist, dtype=bytes).astype(str)
    elif opts['directory'] is not None and opts['filelist'] is None:
        fname = buildFileList(opts['directory'])
        filelist = np.loadtxt(fname, dtype=bytes).astype(str)
    else:
        assert False, 'Must provide input directory or file list'

    # Initialize engine for SQL database
    engine = Engine(opts['dbname'], opts['dbtype'])

    # Initialize an instrument
    inst = instrument.select(opts['type'], fmt=opts['format'])
    # Set its format for reading ASCII data
    inst.updateASCIIformat(opts['format'], columns=opts['columns'])

    # Initialize a network object for handling metadata
    net = network.Network()
    # This will return an empty data frame if no metafile exists
    net.read_metadata_ascii(opts['metafile'], fmtdict=opts['metafmt'])
    meta_df = net.meta_to_df()

    # If database doesn't exist, create it and initialize output SQL tables
    ref_meta = None
    if not os.path.isfile(engine.dbname):
        engine.initdb(inst.create_cmd)
        
    # If it does exist, read the list of files that already exist in the 
    # database and only keep the new ones
    else:
        print('Constructing list of unique files')
        filelist = engine.getUniqueFiles(filelist)
        # Also try to read reference metadata
        try:
            ref_meta = engine.meta()
        except ValueError:
            pass
    print('Number of new files to add to database:', len(filelist))
    
    # Loop over new filenames 
    cnt = 0
    stations = []; longitude = []; latitude = []; elev = []
    for filepath in filelist:

        if cnt % 100 == 0:
            sys.stdout.write(' - %d\r' % cnt)
            sys.stdout.flush()

        # Instrument parses the data to make a data frame
        data_df, statname = inst.readASCII(filepath)

        # Get metadata from file or external source
        #if statname not in meta_df['id'].values:
        #    add_meta = inst.read_meta_header(filepath)
        #    meta_df.append(add_meta, ignore_index=True)
        
        # Add data to SQL database
        engine.addData(data_df)
        # And the file path
        engine.addFile(filepath)
        cnt += 1

    print('')

    # Make unique metadata table
    meta_df = pd.DataFrame({'id': stations, 'lon': longitude, 'lat': latitude, 'elev': elev})
    if ref_meta is None:
        meta_df.to_sql('metadata', engine.engine)
    else:
        meta_df = pd.concat([meta_df, ref_meta]).drop_duplicates(subset='id')
        meta_df.to_sql('metadata', engine.engine, if_exists='replace')


# end of file
