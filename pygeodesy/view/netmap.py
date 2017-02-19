#-*- coding: utf-8 -*_

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

from ..db.Engine import Engine
import pygeodesy.instrument as instrument
import pygeodesy.network.Network as Network
from .plot import model_and_detrend

# Define the default options
defaults = {
    'input': 'sqlite:///data.db',
    'statlist': None,
    'tstart': None,
    'tend': None,
    'ylim': (None, None),
    'model': 'filt',
    'station_labels': True,
    'coefficients': False,
}


def netmap(optdict):

    # Update the options
    opts = defaults.copy()
    opts.update(optdict)

    # Map matplotlib color codes to the seaborn palette
    try:
        import seaborn as sns
        sns.set_color_codes()
    except ImportError:
        pass

    # Create engine for input database
    engine = Engine(url=opts['input'])

    # Initialize an instrument
    inst = instrument.select(opts['type'])
    components = engine.components()

    # Make a network object
    network = Network(inst, engine)
    print(' - %d stations' % network.nstat)
   
    # Read date array
    dates = engine.dates()

    # Determine plotting bounds
    tstart = np.datetime64(opts['tstart']) if opts['tstart'] is not None else None
    tend = np.datetime64(opts['tend']) if opts['tend'] is not None else None

    # Get a list of unique station roots
    roots = [name.split('_')[0] for name in network.names]

    # Make a map of the station
    fig1, ax_map = plt.subplots()
    fig2, ax_ts = plt.subplots(nrows=len(components))
    bmap = network.makeBasemap(ax_map, plot_stat=True, station_labels=True)
    sx, sy = bmap(network.lon, network.lat)

    if type(ax_ts) not in (list, np.ndarray):
        ax_ts = [ax_ts]
    
    # Define function to plot selected time series
    def plot_selected_tseries(event):
        """
        Define what happens when user clicks on the station map.
        """
        # Find close stations
        dist = np.sqrt((sy - event.ydata)**2 + (sx - event.xdata)**2)
        ind_close = (dist < 10000.0).nonzero()[0]
        names_close = network.names[ind_close]
        if len(ind_close) > 0:
            # Find closest one
            ind_closest = np.argmin(dist[ind_close])
            name_closest = names_close[ind_closest]
            print('Plotting station', name_closest)
            # Read data for nearest station and for each component
            ax_ts[0].set_title(name_closest)
            for ax, comp in zip(ax_ts, components):

                ax.cla()
                if not opts['coefficients']:

                    # Read raw data
                    data = network.get(comp, name_closest, with_date=False)
                    data = data.values.squeeze()

                    # Try to read model data
                    fit = model_and_detrend(data, engine, name_closest, comp, opts['model'])

                    # Remove means
                    dat_mean = np.nanmean(data)
                    data -= dat_mean
                    fit -= dat_mean

                    # Compute variance of data if possible
                    try:
                        resid = data - fit
                        std = np.nanstd(resid)
                        mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
                        print(' - std:', std)
                        print(' - mad:', mad)
                    except:
                        pass

                    # Plot time series
                    ax.plot(network.tdec, data, 'o', alpha=0.5)
                    ax.plot(network.tdec, fit, '-r', linewidth=2)

                else:
                    coeff = network.get('coeff_%s' % comp, name_closest, with_date=False)
                    ax.plot(coeff, 'o')
                ax.set_ylabel(comp)
                ax.tick_params(labelsize=14)

        ax_ts[0].set_title(name_closest, fontsize=18)
        ax_ts[-1].set_xlabel('Year')
        fig2.canvas.draw()

    cid = fig1.canvas.mpl_connect('button_press_event', plot_selected_tseries)
    plt.show()
    fig1.canvas.mpl_disconnect(cid)



# end of file
