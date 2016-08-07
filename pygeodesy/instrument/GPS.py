#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .Station import STN
from scipy.spatial import cKDTree
import SparseConeQP as sp
from matutils import dmultl, dmultr
from timeutils import generateRegularTimeArray
import datetime as dtime
import tsinsar as ts
import sys, os

from .TimeSeries import TimeSeries
from .utils import xyz2llh

class GPS(TimeSeries):
    """
    Class to hold GPS station data and perform inversions
    """
    
    # Default column settings
    columns = {'east': 0, 'north': 1, 'up': 2, 'sigma_east': 3, 'sigma_north': 4,
        'sigma_up': 5, 'year': None, 'month': None, 'day': None, 'hour': None, 
        'doy': None}


    def __init__(self, name='gps', stnfile=None, datformat=None, **kwargs):
        """
        Initiate GPS structure with station list
        """
        # Initialize the parent class
        super().__init__(name=name, stnfile=stnfile, dtype='gps', **kwargs)

        self.datformat = datformat
        return


    def updateASCIIformat(self, fmt, columns=None):
        """
        Set the column format from either a pre-supported format or
        a user-provided dictionary of the column format.
        """
 
        new_columns = None

        # Read a supported format        
        if fmt is not None:
            if fmt == 'gipsy':
                new_columns = {'east': 0, 'north': 1, 'up': 2, 'sigma_east': 3,
                    'sigma_north': 4, 'sigma_up': 5, 'year': 9, 'month': 10,
                    'day': 11, 'hour': 12}
                self.read_header = True
            elif fmt == 'sopac':
                new_columns = {'year': 1, 'doy': 2, 'north': 3, 'east': 4, 'up': 5,
                    'sigma_north': 6, 'sigma_east': 7, 'sigma_up': 8}

        # Or parse 'columns' string to make a dictionary
        elif columns is not None:
            new_columns = {}
            fields = columns.strip('{}').split(',')
            for keyval_str in columns.strip('{}').split(','):
                key, value = keyval_str.split(':')
                new_columns[key.strip()] = int(value)

        # Update the columns dictionary
        if new_columns is not None:
            self.columns.update(new_columns)

        return


    def read_meta_header(self, filename, meta_dict=None):
        """
        Try to read metadata from header for gipsy format files.
        """
        # Only for gipsy
        if self.datformat is None or self.datformat != 'gipsy':
            return

        # Get the cartesian coordinates from the header
        with open(filename, 'r') as ifid:
            for line in ifid:
                if 'STA X' in line:
                    statX = float(line.split()[5])
                elif 'STA Y' in line:
                    statY = float(line.split()[5])
                elif 'STA Z' in line:
                    statZ = float(line.split()[5])
                elif 'SRGD' in line:
                    break

        # Convert to lat/lon/h
        lat,lon,h = xyz2llh(np.array([statX, statY, statZ]), deg=True)

        # Update or return dictionary
        statname = filename.split('/')[-1][:4].lower()
        if meta_dict is not None:
            if not statname in meta_dict['id']:
                meta_dict['id'].append(statname)
                meta_dict['lon'].append(lon)
                meta_dict['lat'].append(lat)
                meta_dict['elev'].append(h)
            return
        else:
            return {'id': statname, 'lon': lon, 'lat': lat, 'elev': h}

    
    def read_data(self, gpsdir, fileKernel='CleanFlt', dataFactor=1000.0):
        """
        Reads GPS data and convert to mm
        """
        self.stns = []
        lon = []; lat = []; elev = []; name = []
        # If the list of stations/coordinates have already been set, loop over them
        if self.nstat > 0:
            for ii in range(self.nstat):
                stn = STN(self.name[ii], gpsdir, format=self.datformat,
                    fileKernel=fileKernel, dataFactor=dataFactor)
                if stn.success:
                    self.stns.append(stn)
                    lon.append(self.lon[ii])
                    lat.append(self.lat[ii])
                    elev.append(self.elev[ii])
                    name.append(self.name[ii])

        # If not, we loop over the files listed in gpsdir, and pull the coordinates
        # from the header
        else:
            for root, dirs, files in os.walk(gpsdir):
                for fname in files:
                    if fileKernel not in fname:
                        continue
                    statname = fname[:4].lower()
                    stn = STN(statname, gpsdir, format=self.datformat,
                        fileKernel=fileKernel, dataFactor=dataFactor, getcoords=True)
                    self.stns.append(stn)
                    lon.append(stn.lon)
                    lat.append(stn.lat)
                    elev.append(stn.elev)
                    name.append(statname)
                   
        # Save coordinates and names to self 
        for key, value in (('lon', lon), ('lat', lat), ('elev', elev), ('name', name)):
            setattr(self, key, value)
        self.nstat = len(name)
        return


    def preprocess(self, hdrformat='filt', dataFactor=1000.0):
        """
        Preprocess the data to remove any known offsets as listed in the 
        Sopac header file. Only for Sopac data formats
        """

        # Don't do anything if we're not using Sopac
        if self.datformat != 'sopac':
            return

        if hdrformat == 'filt':
            csopac = ts.sopac.sopac
        elif hdrformat == 'trend':
            csopac = sopac

        # Loop through stations
        print('Preprocessing for realz')
        for stn in self.stns:
            smodel = csopac(stn.fname)
            components = [smodel.north, smodel.east, smodel.up]
            data = [stn.north, stn.east, stn.up]
            # Loop through components
            for ii in range(len(data)):
                comp = components[ii]
                # Get representation for offset
                frep = comp.offset
                if len(frep) < 1:
                    continue
                rep = []; amp = []
                for crep in frep:
                    print(stn.fname, crep.rep, crep.amp)
                    rep.append(crep.rep)
                    amp.append(crep.amp)
                # Construct design matrix
                plt.plot(stn.tdec, data[ii], 'o')
                G = np.asarray(ts.Timefn(rep, stn.tdec)[0], order='C')
                amp = dataFactor * np.array(amp)
                # Compute modeled displacement and remove from data
                fit = np.dot(G, amp)
                data[ii] -= fit
                #plt.plot(stn.tdec, data[ii], 'or')
                #plt.show()


# end of file
