#-*- coding: utf-8 -*_

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dtime
import sys
import os

from ..db.Engine import Engine
import pygeodesy.instrument as instrument
import pygeodesy.network.Network as Network

# Define the default options
defaults = {
    'input': 'sqlite:///data.db',
    'index': None,
    'figwidth': 12,
    'figheight': 7,
    'scale': None,
    'quiverkey': None,
    'model': None,
}


def velmap(optdict):

    # Update the options
    opts = defaults.copy()
    opts.update(optdict)

    # Map matplotlib color coes to the seaborn palette
    try:
        import seaborn as sns
        sns.set_color_codes()
    except ImportError:
        pass

    # Create engine for input database
    engine = Engine(url=opts['input'])

    # Initialize an instrument
    inst = instrument.select(opts['type'])

    # Make a network object
    network = Network(inst, engine)
    print(' - %d stations' % network.nstat)

    # Read metadata
    meta = engine.meta()

    # Build the velocity arrays
    east = []; north = []
    if opts['model'] is not None:

        for statname in meta['id']:
            print(statname)
            east_df = network.get('%s_east' % opts['model'], statname)
            north_df = network.get('%s_north' % opts['model'], statname)

            # Get slopes
            phi_east = np.polyfit(network.tdec, east_df.values, 1)
            phi_north = np.polyfit(network.tdec, north_df.values, 1)
            east.append(phi_east[0])
            north.append(phi_north[0])

    elif opts['index'] is not None:

        index = int(opts['index'])
        for statname in meta['id']:
            east_df = network.get('coeff_east', statname)
            north_df = network.get('coeff_north', statname)
            east.append(east_df.values[index])
            north.append(north_df.values[index])

    elif opts['window'] is not None:

        # Parse the window string to get a start and end time
        tstart, tend = opts['window'].split(',')
        tstart = dtime.datetime.strptime(tstart.strip(), '%Y-%m-%d')
        tend = dtime.datetime.strptime(tend.strip(), '%Y-%m-%d')

        # Make a time mask
        tind = (network.dates > tstart) * (network.dates < tend)
        tsub = network.tdec[tind].squeeze()

        # Loop over stations
        for statname in meta['id']:
            print(statname)
            east_df = network.get('east', statname)
            north_df = network.get('north', statname)

            east_data = east_df.values[tind].squeeze()
            north_data = north_df.values[tind].squeeze()
            finite = np.isfinite(east_data) * np.isfinite(north_data)
            nfinite = len(finite.nonzero()[0])
            if nfinite < 10:
                east.append(np.nan)
                north.append(np.nan)
            else:
                phi_east = np.polyfit(tsub[finite], east_data[finite], 1)
                phi_north = np.polyfit(tsub[finite], north_data[finite], 1)
                east.append(phi_east[0])
                north.append(phi_north[0])
                

    if opts['quiverkey'] is not None:
        qkey = float(opts['quiverkey'])
    else:
        amp = np.sqrt(np.array(east)**2 + np.array(north)**2)
        qkey = 2*np.median(amp)

    # Make a map of the station
    figsize = (int(opts['figwidth']), int(opts['figheight']))
    fig1, ax_map = plt.subplots(figsize=figsize)
    bmap = network.makeBasemap(ax_map)

    scale = float(opts['scale']) if opts['scale'] is not None else None
    q = bmap.quiver(meta['lon'].values, meta['lat'].values, east, north, 
        latlon=True, scale=scale)
    plt.quiverkey(q, 0.9, 0.1, qkey, '%4.2f' % qkey, coordinates='axes')
    plt.show()


# end of file
