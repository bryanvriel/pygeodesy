#-*- coding: utf-8 -*_

import numpy as np
import pandas as pd
import pygeodesy as pg
import pyre
import matplotlib.pyplot as plt
import sys
import os

class Plot(pg.components.task, family='pygeodesy.plot'):
    """
    Plot component time series.
    """
    
    input = pyre.properties.str(default='sqlite:///data.db')
    input.doc = 'Input time series database'

    component = pyre.properties.str(default='all')
    component.doc = 'Deformation component to model (default: all)'

    residual = pyre.properties.bool(default=False)
    residual.doc = 'Plot (data - model) (default: False)'

    stations = pyre.properties.str(default=None)
    stations.doc = 'List of stations to plot'

    save = pyre.properties.bool(default=False)
    save.doc = 'Flag to save figure to file (default: False)'

    tstart = pyre.properties.float(default=None)
    tstart.doc = 'Starting decimal year for plot'

    tend = pyre.properties.float(default=None)
    tend.doc = 'Ending decimal year for plot'

    ylim = pyre.properties.str(default=None)
    ylim.doc = 'Y-limit for plot'

    model = pyre.properties.str(default='full')
    model.doc = 'Model component to plot (secular, seasonal, transient, step, full)'

    output_dir = pyre.properties.str(default='figures')
    output_dir.doc = 'Directory for saving figures'

    figwidth = pyre.properties.int(default=10)
    figwidth.doc = 'Figure width'

    figheight = pyre.properties.int(default=6)
    figheight.doc = 'Figure height'

    kml = pyre.properties.str(default=None)
    kml.doc = 'Output KML file for station coordinates'

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

        # Plot KML if requested and exit
        if self.kml is not None:
            from .kml import make_kml
            make_kml(engine, self.kml)
            return

        # Get the list of stations to plot
        if self.stations == 'all':
            import pygeodesy.network.Network as Network
            network = Network(inst, engine)
            statnames = network.names
        else: 
            statnames = self.stations.split()

        # Read data after checking for existence of component
        if self.component == 'all':
            components = engine.components()
        elif self.component not in engine.components():
            components = [engine.components()[0]]
        else:
            components = [self.component]

        # Read data array
        dates = engine.dates()

        # Determine plotting bounds
        tstart = np.datetime64(self.tstart) if self.tstart is not None else None
        tend = np.datetime64(self.tend) if self.tend is not None else None

        # Set the figure size
        figsize = (self.figwidth, self.figheight) 
        fig, axes = plt.subplots(nrows=len(components), figsize=figsize)
        if type(axes) not in (list, np.ndarray):
            axes = [axes]

        # Check output directory exists if we're saving
        if self.save and not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        # Loop over stations
        for statname in statnames:

            # Check if we had previously closed the figure and need to make new one
            if fig is None and axes is None:
                fig, axes = plt.subplots(nrows=len(components), figsize=figsize)
                if type(axes) not in (list, np.ndarray):
                    axes = [axes]

            print(statname)

            for ax, component in zip(axes, components):

                # Read data
                data = pd.read_sql_table(component, engine.engine, columns=[statname,])
                data = data[statname].values.squeeze()
                
                # Try to read model data
                fit = pg.view.utils.model_and_detrend(data, engine, statname,
                                                      component, self.model)

                # Remove means
                dat_mean = np.nanmean(data)
                data -= dat_mean
                fit -= dat_mean
                residual = data - fit

                # Plot residuals
                if self.residual:
                    std = np.nanmedian(np.abs(residual - np.nanstd(residual)))
                    residual[np.abs(residual) > 4*std] = np.nan
                    line, = ax.plot(dates, residual, 'o', alpha=0.6, zorder=10)

                # Or data and model
                else:
                    line, = ax.plot(dates, data, 'o', alpha=0.6, zorder=10)
                    ax.plot(dates, fit, '-r', linewidth=6, zorder=11)

                    # Also try to read "raw" data (for CME results)
                    try:
                        raw = pd.read_sql_table('raw_' + component, 
                            engine.engine, columns=[statname,])
                        raw = raw.values.squeeze() - dat_mean
                        ax.plot(dates, raw, 'sg', alpha=0.7, zorder=9)
                    except:
                        pass

                ax.tick_params(labelsize=18)
                ax.set_ylabel(component, fontsize=18)
                ax.set_xlim(tstart, tend)
                if self.ylim is not None:
                    y0, y1 = [float(val) for val in self.ylim.split(',')]
                    ax.set_ylim(y0, y1)

            axes[0].set_title(statname, fontsize=18)
            axes[-1].set_xlabel('Year', fontsize=18)
            
            if self.save:
                plt.savefig('%s/%s_%s.png' % (self.output_dir, statname, component), 
                    dpi=200, bbox_inches='tight')
                fig = axes = None
            else:
                plt.show()
            plt.close('all') 


# end of file
