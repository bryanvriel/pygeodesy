#-*- coding: utf-8 -*_

import pygeodesy as pg
import pyre
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

class NetMap(pg.components.task, family='pygeodesy.netmap'):
    """
    Makes interactive map of station network.
    """

    input = pyre.properties.str(default='sqlite:///data.db')
    input.doc = 'Input time series database'
    
    model = pyre.properties.str(default='filt')
    model.doc = 'Model component to plot'

    coefficients = pyre.properties.bool(default=False)
    coefficients.doc = 'Flag to plot time series model coefficients when clicking on station'

    @pyre.export
    def main(self, plexus, argv):
        """
        Main entrypoint into this application.
        """

        # Map matplotlib color codes to the seaborn palette
        try:
            import seaborn as sns
            sns.set_color_codes()
        except ImportError:
            pass

        # Create engine for input database
        engine = pg.db.Engine(url=self.input)

        # Initialize an instrument
        inst = pg.instrument.select(plexus.data_type)
        components = engine.components()

        # Make a network object
        network = pg.network.Network(inst, engine)
        print(' - %d stations' % network.nstat)
   
        # Read date array
        dates = engine.dates()

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
                    if not self.coefficients:

                        # Read raw data
                        data = network.get(comp, name_closest, with_date=False)
                        data = data.values.squeeze()

                        print(name_closest, data)

                        # Try to read model data
                        fit = pg.view.utils.model_and_detrend(data, engine, name_closest,
                                                              comp, self.model)

                        # Remove means
                        #dat_mean = np.nanmean(data)
                        #data -= dat_mean
                        #fit -= dat_mean

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
