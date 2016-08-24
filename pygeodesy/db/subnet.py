#-*- coding: utf-8 -*_

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy_utils import create_database
import pandas as pd
import sys
import os

from .Engine import Engine
from .Interface import Interface
import pygeodesy.instrument as instrument

# Define the default options
defaults = {
    'input': 'sqlite:///data.db',
    'output': 'sqlite:///sub.db',
    'poly': None,
    'list': None,
    'tstart': None,
    'tend': None,
}


def subnet(optdict):

    # Update the options
    opts = defaults.copy()
    opts.update(optdict)

    # Create engine for input database
    engine = Engine(url=opts['input'])

    # And for output engine
    engine_out = Engine(url=opts['output'])
    # Also initialize it
    engine_out.initdb(new=True)

    # Initialize an instrument
    inst = instrument.select(opts['type'])

    # Make an interface object to link the instrument and SQL table
    interface = Interface(inst, engine)

    # Read metadata from input table
    meta = engine.meta()
    names = meta['id'].values

    # Get list of files 
    files = engine.getUniqueFiles()

    # Use polynomial for a mask
    if opts['poly'] is not None:

        # Cache the raw lon/lat values
        lon, lat = meta['lon'].values, meta['lat'].values

        # Load points from polynomial file
        from matplotlib.path import Path
        plon, plat = np.loadtxt(opts['poly'], unpack=True)
    
        # Make a path object to compute mask
        poly = Path(np.column_stack((plon, plat)))
        mask = poly.contains_points(list(zip(lon, lat)))

        # Subset stations
        stations = names[mask]

    elif opts['list'] is not None:
        
        # Read stations
        input_stations = np.loadtxt(opts['list'], dtype=bytes).astype(str)

        # Keep ones that are in the database
        stations = np.intersect1d(input_stations, names)

    # Subset metadata and write to table
    meta_sub = meta[np.in1d(names, stations)].reset_index(drop=True)
    meta_sub.to_sql('metadata', engine_out.engine, if_exists='replace')

    # Subset the data table using station list
    interface.subset_table(stations, engine_out, tstart=opts['tstart'], 
        tend=opts['tend'], filelist=files)
    

# end of file
