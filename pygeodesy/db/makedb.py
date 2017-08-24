#-*- coding: utf-8

from __future__ import print_function
from .Engine import Engine
from .Interface import Interface
from .dbutils import *
import pandas as pd
import datetime
import sys
import os

import pygeodesy.instrument as instrument

# Define the default options
defaults = {
    'columns': None,
    'directory': None,
    'dbname': 'data.db',
    'dbtype': 'sqlite',
    'filelist': None,
    'metafile': None,
    'metafmt': None,
    'chunk_size': 100000,
    'extension': '.neu',
    'preprocess': False,
    'fixed_widths': None,
}


def makedb(optdict):

    # Update the options
    opts = defaults.copy()
    opts.update(optdict)

    # Read list of files or build
    if opts['directory'] is None and opts['filelist'] is not None:
        filelist = np.loadtxt(opts['filelist'], dtype=bytes).astype(str)
    elif opts['directory'] is not None and opts['filelist'] is None:
        fname = buildFileList(opts['directory'], opts['format'], opts['extension'])
        filelist = np.loadtxt(fname, dtype=bytes).astype(str)
    else:
        assert False, 'Must provide input directory or file list'

    # Initialize engine for SQL database
    engine = Engine(dbname=opts['dbname'], dbtype=opts['dbtype'])

    # Initialize an instrument
    inst = instrument.select(opts['type'], fmt=opts['format'])
    # Set its format for reading ASCII data
    inst.updateASCIIformat(opts['format'], columns=opts['columns'])
    # Parse any fixed width strings
    inst.parseFixedWidths(opts['fixed_widths'])

    # Read a metadata file if provided
    inst.read_metadata_ascii(opts['metafile'], fmtdict=opts['metafmt'])
    meta_dict = inst.reformat_metadata(fmt='dict')

    # Initialize the database
    engine.initdb(new=False)

    # Get unique file list
    filelist = engine.getUniqueFiles(newlist=filelist)
    print('Number of new files to add to database:', len(filelist))

    # Make an interface object to link the instrument and SQL table
    interface = Interface(inst, engine)
    interface.data_to_table(filelist, meta_dict, chunk_size=int(opts['chunk_size']))

    # Update metadata
    interface.update_meta(meta_dict)
    

# end of file
