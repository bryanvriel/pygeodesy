#-*- coding: utf-8 -*_

import pygeodesy as pg
import pyre
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dtime
import sys
import os

class VelMap(pg.components.task, family='pygeodesy.velmap'):
    """
    Plot vector map of displacements.
    """

    input = pyre.properties.str(default='sqlite:///data.db')
    input.doc = 'Input time series database'

    figwidth = pyre.properties.int(default=10)
    figwidth.doc = 'Figure width'

    figheight = pyre.properties.int(default=6)
    figheight.doc = 'Figure height'

    coeff_index = pyre.properties.int(default=None)
    coeff_index.doc = 'Coefficient index to plot as vector components'

    scale = pyre.properties.float(default=1.0)
    scale.doc = 'Quiver scale factor'

    quiverkey = pyre.properties.float(default=None)
    quiverkey.doc = 'Reference vector magnitude for quiver legend'

    model = pyre.properties.str(default=None)
    model.doc = 'Model component to plot (secular, seasonal, transient, step, full, filt)'

    window = pyre.properties.str(default=None)
    window.doc = 'Temporal window to compute displacements [YYYY-MM-DD, YYYY-MM-DD]'

    @pyre.export
    def main(self, plexus, argv):
        """
        Main entrypoint into this application.
        """

        # Map matplotlib color coes to the seaborn palette
        try:
            import seaborn as sns
            sns.set_color_codes()
        except ImportError:
            pass

        # Create engine for input database
        engine = pg.db.Engine(url=self.input)

        # Initialize an instrument
        inst = pg.instrument.select(plexus.data_type)

        # Make a network object
        network = pg.network.Network(inst, engine)
        print(' - %d stations' % network.nstat)

        # Read metadata
        meta = engine.meta()

        # Parse the window string to get a start and end time
        if self.window is not None:

            # Make datetimes
            tstart, tend = self.window.split(',')
            tstart = dtime.datetime.strptime(tstart.strip(), '%Y-%m-%d')
            tend = dtime.datetime.strptime(tend.strip(), '%Y-%m-%d')

            # Make a time mask
            tmask = (network.dates > tstart) * (network.dates < tend)
        else:
            tmask = np.ones(network.tdec.size, dtype=bool)

        # Build the velocity arrays
        east = []; north = []

        # Use modeled time series
        if self.model is not None:

            for statname in meta['id']:

                # Get modeled displacement data
                print(statname)
                east_df = network.get('%s_east' % self.model, statname)
                north_df = network.get('%s_north' % self.model, statname)

                # Mask out NaN values
                mask = (tmask * np.isfinite(east_df.values.squeeze()) *
                        np.isfinite(north_df.values.squeeze()))

                # Fit polynomial (1st-order)
                phi_east = np.polyfit(network.tdec[mask], east_df.values[mask], 1)
                phi_north = np.polyfit(network.tdec[mask], north_df.values[mask], 1)
                east.append(phi_east[0])
                north.append(phi_north[0])

        # Or coefficient indices
        elif self.coeff_index is not None:

            for statname in meta['id']:
                east_df = network.get('coeff_east', statname)
                north_df = network.get('coeff_north', statname)
                east.append(east_df.values[self.coeff_index])
                north.append(north_df.values[self.coeff_index])

        # Or displacement data
        else:
        
            # Loop over stations
            for statname in meta['id']:

                # Get observed displacement data
                print(statname)
                if statname in ['pcal']: continue
                east_df = network.get('east', statname)
                north_df = network.get('north', statname)

                # Mask out NaN values
                mask = (tmask * np.isfinite(east_df.values.squeeze()) *
                        np.isfinite(north_df.values.squeeze()))

                # Fit polynomial (1st-order)
                east_data = east_df.values[mask].squeeze()
                north_data = north_df.values[mask].squeeze()
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

        if self.quiverkey is not None:
            qkey = self.quiverkey
        else:
            amp = np.sqrt(np.array(east)**2 + np.array(north)**2)
            qkey = 2*np.median(amp)

        # Make a map of the station
        figsize = (self.figwidth, self.figheight)
        fig1, ax_map = plt.subplots(figsize=figsize)
        bmap = network.makeBasemap(ax_map)

        for name, lon, lat in zip(meta['id'].values, meta['lon'].values, meta['lat'].values):
            sx, sy = bmap(lon, lat)
            plt.annotate(name, xy=(sx,sy))

        scale = self.scale if self.scale is not None else None
        q = bmap.quiver(meta['lon'].values, meta['lat'].values, east, north, 
                        latlon=True, scale=scale)
        plt.quiverkey(q, 0.9, 0.1, qkey, '%4.2f' % qkey, coordinates='axes')
        plt.show()


# end of file
