#-*- coding: utf-8 -*_

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

from ..db.Engine import Engine
import pygeodesy.instrument as instrument

# Define the default options
defaults = {
    'input': 'sqlite:///data.db',
    'component': None,
    'stations': None,
    'statlist': None,
    'save': False,
    'tstart': None,
    'tend': None,
    'ylim': (None, None),
    'model': 'filt',
    'figwidth': 10,
    'figheight': 6,
}


def plot(optdict):

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

    # Get the list of stations to plot
    statnames = opts['stations'].split()

    # Read data after checking for existence of component
    if opts['component'] not in engine.components():
        component = engine.components()[0]
    else:
        component = opts['component']
    model = opts['model']
    model_comp = '%s_%s' % (model, component) if model != 'filt' else 'None'
    filt_comp = 'filt_' + component

    # Read data array
    dates = engine.dates()

    # Determine plotting bounds
    tstart = np.datetime64(opts['tstart']) if opts['tstart'] is not None else None
    tend = np.datetime64(opts['tend']) if opts['tend'] is not None else None

    # Determine y-axis bounds
    if type(opts['ylim']) is str:
        y0, y1 = [float(y) for y in opts['ylim'].split(',')]
    else:
        y0, y1 = opts['ylim']

    # Set the figure size
    figsize = (int(opts['figwidth']), int(opts['figheight']))

    # Get list of tables in the database
    tables = engine.tables(asarray=True)

    # Construct list of model components to remove (if applicable)
    if model == 'secular':
        parts_to_remove = ['seasonal', 'transient']
    elif model == 'seasonal':
        parts_to_remove = ['secular', 'transient']
    elif model == 'transient':
        parts_to_remove = ['secular', 'seasonal']
    elif model == 'full':
        parts_to_remove = []

    # Loop over stations
    for statname in statnames:

        # Read data
        data = pd.read_sql_query("SELECT %s FROM %s;" % (statname, component),
            engine.engine)
        data = data[statname].values.squeeze()

        # Try to read model data
        fit = np.nan * np.ones_like(data)
        if model_comp in tables:

            # Read full model data
            fit = pd.read_sql_query("SELECT %s FROM full_%s;" % (statname, component),
                engine.engine).values.squeeze()

            # Remove parts we do not want
            for ftype in parts_to_remove:
                signal = pd.read_sql_query("SELECT %s FROM %s_%s;" % (statname,
                    ftype, component), engine.engine).values.squeeze()
                fit -= signal
                data -= signal

        elif filt_comp in tables:
            fit = pd.read_sql_query("SELECT %s FROM filt_%s;" % (statname, component),
                engine.engine).values.squeeze()

        # Remove means
        dat_mean = np.nanmean(data)
        data -= dat_mean
        fit -= dat_mean

        # Plot data
        fig, ax = plt.subplots(figsize=figsize)
        line, = ax.plot(dates, data, 'o', alpha=0.6)
        ax.plot(dates, fit, '-r', linewidth=6)
        #ax.plot(dates, data - fit, 'o', alpha=0.6)
        ax.tick_params(labelsize=18)
        ax.set_xlabel('Year', fontsize=18)
        ax.set_ylabel(component, fontsize=18)
        ax.set_xlim(tstart, tend)
        ax.set_ylim(y0, y1)
        ax.set_xticks(ax.get_xticks()[::2])
        if opts['save']:
            plt.savefig('%s_%s.png' % (statname, component), 
                dpi=200, bbox_inches='tight')
        plt.show()
        plt.close('all')        


    # Read metadata from input table
    #meta = engine.meta()
    #names = meta['id'].values

    



# end of file
