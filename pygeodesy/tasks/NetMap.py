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
        fig1, ax_map = plt.subplots(figsize=(9,6))
        fig2, ax_ts = plt.subplots(nrows=len(components), figsize=(9,7))
        bmap = network.makeBasemap(ax_map, plot_stat=True, station_labels=True)
        sx, sy = bmap(network.lon, network.lat)

        if type(ax_ts) not in (list, np.ndarray):
            ax_ts = [ax_ts]
        
        # Define function to plot selected time series
        def plot_selected_tseries(event):
            """
            Define what happens when user clicks on the station map.
            """
            # don't do anything if user clicked outside the map boundaries
            if (event.xdata is None) or (event.ydata is None): return
            # Find close stations
            dist = np.sqrt((sy - event.ydata)**2 + (sx - event.xdata)**2)
            ind_close = (dist < 10000.0).nonzero()[0]
            if len(ind_close) > 0:
                # Find closest one
                names_close = network.names[ind_close]
                ind_closest = np.argmin(dist[ind_close])
                name_closest = names_close[ind_closest]
                print('Plotting station', name_closest)
                # Read data for nearest station and for each component
                ax_ts[0].set_title(name_closest)
                for ax, comp in zip(ax_ts, components):
                    print(' - component:', comp)

                    ax.cla()
                    if not self.coefficients:

                        # Read raw data
                        data = network.get(comp, name_closest, with_date=False)
                        data = data.values.squeeze()

                        # Try to read model data
                        fit = pg.view.utils.model_and_detrend(data, engine, name_closest,
                                                              comp, self.model)

                        # Compute statistics of residuals if a model fit was computed
                        fit_finite = np.isfinite(fit).nonzero()[0]
                        if fit_finite.size > 0:
                            resid = data - fit
                            std = np.nanstd(resid)
                            mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
                            print('   - residual std:', std)
                            print('   - residual MAD:', mad)

                        # Otherwise, just print out statistics of original time series
                        else:
                            std = np.nanstd(data)
                            mad = np.nanmedian(np.abs(data - np.nanmedian(data)))
                            print('   - std:', std)
                            print('   - MAD:', mad)

                        # Plot time series
                        ax.plot(network.tdec, data, 'o', alpha=0.5)
                        ax.plot(network.tdec, fit, '-r', linewidth=2)

                    else:
                        coeff = network.get('coeff_%s' % comp, name_closest, with_date=False)
                        ax.plot(coeff, 'o', fontsize=12)
                    ax.set_ylabel(comp)
                    ax.tick_params(labelsize=12)

                ax_ts[0].set_title(name_closest, fontsize=14)
                ax_ts[-1].set_xlabel('Year', fontsize=12)
                fig2.canvas.draw()

        cid = fig1.canvas.mpl_connect('button_press_event', plot_selected_tseries)
        plt.show()
        fig1.canvas.mpl_disconnect(cid)



# end of file
