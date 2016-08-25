#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import datetime as dtime
import tsinsar as ts
import h5py
import sys
import os

from ..utilities import datestr2tdec
from .TimeSeries import TimeSeries

class Wells(TimeSeries):
    """
    Class to hold well stations.
    """

    # The string for inserting data into a SQL table
    insert_cmd = ("INSERT INTO tseries(DATE, up, sigma_up, id) "
           "VALUES(?, ?, ?, ?);")

    # The string for creating an SQL table
    create_cmd = ("CREATE TABLE tseries("
        "DATE TEXT, "
        "up FLOAT, "
        "sigma_up FLOAT, "
        "id TEXT);")


    def __init__(self, name='wells', stnfile=None, stnlist=None, **kwargs):
        """
        Initiate EDM class.
        """
        super().__init__(name=name, stnfile=stnfile, dtype='wells', **kwargs)
        return
        

    def read_data(self, ddir):
        """
        Read the data from the source directory 'ddir'.
        """
        self.statDict = {}
        for root, dirs, files in os.walk(ddir):
            for fname in files:

                # Skip if not an h5 file
                if not fname.endswith('.h5'):
                    continue
                filepath = os.path.join(root, fname)
                fid = h5py.File(filepath, 'r')
                statname = fname.split('.')[0].lower()

                # Get list of groups and iterate
                for groupname, group in fid.items():
                    if '_depth' in groupname: continue
                    # Empty station dictionary
                    stat = {}
                    # Load data
                    datestr = group['dates'].value.astype(str)
                    data = group['ts'].value
                    range_vals = group['range'].value
                    # Convert data to meters
                    data *= 0.3048
                    # Make decimal year for each epoch
                    tdec = []
                    for dstr in datestr:
                        x1, x2 = dstr.split()
                        month, day, year = [int(s) for s in x1.split('/')]
                        if year > 50:
                            year += 1900
                        else:
                            year += 2000
                        hour, minute = [int(s) for s in x2.split(':')]
                        date = dtime.datetime(year, month, day, hour, minute)
                        tdec.append(datestr2tdec(pydtime=date))
                    # Save
                    stat['up'] = data
                    stat['w_up'] = np.ones_like(data)
                    stat['tdec'] = np.array(tdec)
                    stat['midpoint'] = 0.3048*np.mean(range_vals)
                    self.statDict['%s_%s' % (statname, groupname.lower())] = stat

        # Finally, have the statDict remember that this is a 'wells' dtype
        self.statDict['dtype'] = 'wells'
        # Make a station generator
        self.makeStatGen() 

        return


    def read_locations(self, filename):
        """
        Reads the well coordinates from filename.
        """
        # Load the data
        names = np.loadtxt(filename, usecols=(0,), dtype=bytes).astype(str)
        lats, lons = np.loadtxt(filename, usecols=(1,2), unpack=True)
        location_dict = {name.lower(): (lon, lat) for (name, lon, lat) in zip(names, lons, lats)}
        # Save to station dictionary
        for statname, stat in self.statGen:
            wellname = statname.split('_')[0]
            stat['lon'], stat['lat'] = location_dict[wellname]
            stat['elev'] = 0.0
        return 


# end of file
