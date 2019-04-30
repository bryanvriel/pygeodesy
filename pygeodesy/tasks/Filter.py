#-*- coding: utf-8 -*-

import pyre
import pygeodesy as pg

class Filter(pg.components.task, family='pygeodesy.filter'):
    """
    Apply low-pass/smoothing filter to time series database.
    """

    input = pyre.properties.str(default='sqlite:///data.db')
    input.doc = 'Input time series database'

    output = pyre.properties.str(default='sqlite:///filtered.db')
    output.doc = 'Output filtered time series database'

    remove_outliers = pyre.properties.bool(default=False)
    remove_outliers.doc = 'Flag to remove outliers during filtering (default: False)'

    mask = pyre.properties.bool(default=False)
    mask.doc = 'Flag to mask out invalid data after filtering (default: False)'

    kernel_size = pyre.properties.int(default=7)
    kernel_size.doc = 'Kernel size for filter (default: 7)'

    nstd = pyre.properties.int(default=5)
    nstd.doc = 'Number of deviations for outlier threshold (default: 5)'

    std_thresh = pyre.properties.float(default=100.0)
    std_thresh.doc = 'Absolute deviation threshold for bad stations (default: 100.0)'

    deviator = pyre.properties.str(default='std')
    deviator.doc = 'Deviation metric (std, mad) (default: std)'

    log = pyre.properties.bool(default=False)
    log.doc = 'Write info to log file filter.log'

    @pyre.export
    def main(self, plexus, argv):
        """
        Main entrypoint into this application.
        """

        # Create engine for input database
        engine = pg.db.Engine(url=self.input)

        # And for output
        engine_out = pg.db.Engine(url=self.output)
        # Also initialize it
        engine_out.initdb(new=True, ref_engine=engine)

        # Initialize an instrument
        inst = pg.instrument.select(plexus.data_type)

        # Make a network object
        network = pg.network.Network(inst, engine)

        # Filter
        network.filterData(engine_out, self.kernel_size, mask=self.mask,
                           remove_outliers=self.remove_outliers, nstd=self.nstd,
                           std_thresh=self.std_thresh, deviator=self.deviator,
                           log=self.log)


# end of file
