#-*- coding: utf-8 -*_

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

from ..db.Engine import Engine
import pygeodesy.instrument as instrument
import pygeodesy.network.Network as Network

# Define the default options
defaults = {
    'input': 'sqlite:///data.db',
    'index': 0,
    'figwidth': 12,
    'figheight': 7,
    'scale': None,
    'quiverkey': None,
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

    # Make a map of the station
    figsize = (int(opts['figwidth']), int(opts['figheight']))
    fig1, ax_map = plt.subplots(figsize=figsize)
    bmap = network.makeBasemap(ax_map)

    # Read metadata
    meta = engine.meta()

    # Build the velocity arrays
    east = []; north = []
    index = int(opts['index'])
    for statname in meta['id']:
        east_df = network.get('coeff_east', statname)
        north_df = network.get('coeff_north', statname)
        east.append(east_df.values[index])
        north.append(north_df.values[index])

    if opts['quiverkey'] is not None:
        qkey = float(opts['quiverkey'])
    else:
        amp = np.sqrt(np.array(east)**2 + np.array(north)**2)
        qkey = 2*np.median(amp)

    scale = float(opts['scale']) if opts['scale'] is not None else None
    q = bmap.quiver(meta['lon'].values, meta['lat'].values, east, north, 
        latlon=True, scale=scale)
    plt.quiverkey(q, 0.9, 0.1, qkey, '%4.2f' % qkey, coordinates='axes')
    plt.show()


# end of file
