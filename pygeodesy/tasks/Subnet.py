#-*- coding: utf-8 -*_

import pygeodesy as pg
import pyre
import numpy as np
import sys

class Subnet(pg.components.task, family='pygeodesy.subnet'):
    """
    Subset a time series database file spatially and temporally.
    """

    input = pyre.properties.str(default='sqlite:///data.db')
    input.doc = 'Input time series database'

    output = pyre.properties.str(default='sqlite:///sub.db')
    output.doc = 'Output subsetted time series database'

    poly = pyre.properties.str(default=None)
    poly.doc = 'Filename of polygon lon/lat coordinates (space delimited)'

    station_list = pyre.properties.str(default=None)
    station_list.doc = 'Filename containing list of stations to select'

    tstart = pyre.properties.float(default=None)
    tstart.doc = 'Starting decimal year of time window'

    tend = pyre.properties.float(default=None)
    tend.doc = 'Ending decimal year of time window'

    scale = pyre.properties.float(default=1.0)
    scale.doc = 'Scale observations by factor'

    @pyre.export
    def main(self, plexus, argv):
        """
        Main entrypoint into this application.
        """

        # Create engine for input database
        engine = pg.db.Engine(url=self.input)

        # And for output engine
        engine_out = pg.db.Engine(url=self.output)
        # Also initialize it
        engine_out.initdb(new=True)

        # Initialize an instrument
        inst = pg.instrument.select(plexus.data_type)

        # Make an interface object to link the instrument and SQL table
        interface = pg.db.Interface(inst, engine)

        # Read metadata from input table
        meta = engine.meta()
        names = meta['id'].values

        # Get list of files 
        files = engine.getUniqueFiles()

        # Use polynomial for a mask
        if self.poly is not None:

            # Cache the raw lon/lat values
            lon, lat = meta['lon'].values, meta['lat'].values

            # Load points from polynomial file
            from matplotlib.path import Path
            plon, plat = np.loadtxt(self.poly, unpack=True)
        
            # Make a path object to compute mask
            poly = Path(np.column_stack((plon, plat)))
            mask = poly.contains_points(list(zip(lon, lat)))

            # Subset stations
            stations = names[mask]

        elif self.station_list is not None:
            
            # Read stations
            input_stations = np.loadtxt(self.station_list, dtype=bytes).astype(str)

            # Keep ones that are in the database
            stations = np.intersect1d(input_stations, names)

        else:
            assert False, 'Must input list of stations or polynomial.'

        # Consistency check between station list and stations in database
        files, stations = pg.db.utils.check_stations_files(files, stations)

        # Subset metadata and write to table
        meta_sub = meta[np.in1d(names, stations)].reset_index(drop=True)
        meta_sub.to_sql('metadata', engine_out.engine, if_exists='replace')

        # Subset the data table using station list
        interface.subset_table(stations, engine_out, tstart=self.tstart, tend=self.tend,
                               filelist=files, scale=self.scale)
        

# end of file
