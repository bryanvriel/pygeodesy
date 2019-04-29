#-*- coding: utf-8 -*-

import pygeodesy as pg
import pyre
import sys

class Clean(pg.components.task, family='pygeodesy.clean'):
    """
    Clean a time series by removing outliers and removing bad values.
    """

    input = pyre.properties.str(default='sqlite:///data.db')
    input.doc = 'Input time series database'

    output = pyre.properties.str(default='sqlite:///cleaned.db')
    output.doc = 'Output subsetted time series database'

    @pyre.export
    def main(self, plexus, argv):
        """
        Main entrypoint into this application.
        """

        # Check if data type supports cleaning
        if plexus.data_format not in ['sopac']:
            print('Format %s not supported. Exiting.' % plexus.data_format)
            return

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

        # Clean
        network.preprocess(engine_out)


# end of file
