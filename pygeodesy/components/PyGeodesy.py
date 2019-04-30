#-*- coding: utf-8 -*-

import pyre

class PyGeodesy(pyre.plexus, family='pygeodesy.global'):
    """
    The pygeodesy executive and application wrapper.
    """

    pyre_namespace = 'pygeodesy'
    from .Action import Action as pyre_action

    data_type = pyre.properties.str(default='gps')
    data_type.doc = 'Type of time series data'

    data_format = pyre.properties.str(default=None)
    data_format.doc = 'Format of time series data'

    def main(self, *args, **kwargs):
        """
        The main entrypoint into the plexus. Print out some messages about data types.
        """
        print('\nPyGeodesy data type:', self.data_type.upper(), '   format:', self.data_format)
        super().main(*args, **kwargs)

    def help(self, **kwargs):
        """
        Embellish the pyre.plexus help by printing out the banner first.
        """
        # Get pygeodesy package
        import pygeodesy
        # Get a channel
        channel = self.info
        # Make some space
        channel.line()
        # Print header
        channel.line(pygeodesy.meta.header)
        # Call plexus help
        super().help(**kwargs)
        

# end of file 
