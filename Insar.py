#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import datetime as dtime
import tsinsar as ts
import h5py
import sys
import os

from timeutils import datestr2tdec
from .TimeSeries import TimeSeries
from .StationGenerator import StationGenerator


class Insar(TimeSeries):
    """
    Class to hold well stations.
    """

    def __init__(self, name='insar', stnfile=None, stnlist=None):
        """
        Initiate Insar class.
        """
        super().__init__(name=name, stnfile=stnfile, dtype='insar')
        return


    def loadStationH5(self, h5file, fileout=None, copydict=False):
        """
        Transfers data from a GIAnT formatted stack to self.
        """
        self.clear()

        # Read the interferogram data and save to generator
        fid = h5py.File(h5file, 'r')
        if copydict:
            igram = fid['igram'].value
            weights = fid['weights'].value
            self.h5file = {}
        else:
            igram = fid['igram']
            weights = fid['weights']
            self.h5file = fid

        # Get the number of "stations"
        self.nstat = igram[0,:,:].size

        # Instantiate a custom station generator with arrays
        self.statGen = StationGenerator(igram, weights, fid['lat'].value,
            fid['lon'].value, fid['hgt'].value)

        # And make a dictionary that just links to the generator
        self.statDict = self.statGen

        # Also tdec and Jmat
        self.tdec = fid['tdec'].value
        self.Jmat = fid['Jmat'].value
        self.Jmat_raw = fid['Jmat_raw'].value
        if copydict:
            fid.close()

        # Store some geometry parameters
        self.Nifg, self.Ny, self.Nx = self.statDict.los.shape

        return

        
    def loadSeasonalH5(self, h5file):
        """
        Transfers data from an h5py data stack to self. For Insar object, we
        simply save the underlying data array.
        """
        # Open the file and read the data
        self.seasonal_fid = h5py.File(h5file, 'r')
        self.statGen.seasm_los = self.seasonal_fid['seasm_los']
        # Get the number of periodic B-splines
        self.npbspline = seas_dat['npbspline']
        self.npar_seasonal = self.seasonal_fid['seasm_los'].shape[0]
        # Remember that we have seasonal data
        self.have_seasonal = True
        return


# end of file
