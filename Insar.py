#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
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

    def __init__(self, name='insar', stnfile=None, stnlist=None, comm=None):
        """
        Initiate Insar class.
        """
        # Initialize the TimeSeries parent class
        super().__init__(name=name, stnfile=stnfile, dtype='insar')
        # Save some MPI parameters
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.n_workers = self.comm.Get_size()
        # Dictionary mapping dtype to attribute
        self.attr_dict = {'igram': '_igram', 'weight': '_weights',
            'par': '_par', 'recon': '_recon'}
        return


    def loadStationH5(self, h5file, fileout=None, copydict=False):
        """
        Transfers data from a GIAnT formatted stack to self.
        """
        self.clear()

        # Only master worker will have access to underlying data
        self._par = None
        if self.rank == 0:

            # Open the H5 file and store access to igrams or time series and weights
            self.h5file = h5py.File(h5file, 'r')
            try:
                # Load interferograms
                self._igram = self.h5file['igram']
                self._data = self._igram
                self.tdec = self.h5file['tdec'].value
                self.tinsar = self.h5file['tinsar'].value
                self.Jmat = self.h5file['Jmat'].value
            except KeyError:
                # Or load time series
                self._recon = self.h5file['recon']
                self._data = self._recon 
                self.tdec = self.h5file['tdec'].value
            self._weights = self.h5file['weights']

            # Instantiate a custom station generator with arrays
            self.statGen = StationGenerator(self._data, self._weights, 
                self.h5file['lat'].value, self.h5file['lon'].value, self.h5file['elev'].value)

            # And make a dictionary that just links to the generator
            self.statDict = self.statGen

            # Also chunk geometry
            self.chunk_shape = self.h5file['chunk_shape'].value
            self.data_shape = self._data.shape

            # Also try to load parameters for inversion stacks
            try:
                self._par = self.h5file['par']
                self.npar = self._par.shape[0]
            except KeyError:
                pass
            
        else:
            self.tdec = self.Jmat = self.chunk_shape = self.data_shape = None

        # Broadcast some useful variables to the workers
        self.tdec = self.comm.bcast(self.tdec, root=0)
        self.Jmat = self.comm.bcast(self.Jmat, root=0)
        self.chunk_shape = self.comm.bcast(self.chunk_shape, root=0)
        self.data_shape = self.comm.bcast(self.data_shape, root=0)
        self.Ny, self.Nx = self.data_shape[1:]
        self.nstat = self.Ny * self.Nx
        self.Nobs = self.data_shape[0] * self.Ny * self.Nx

        return


    def initialize(self, data_shape, tdec, statGen=None, chunk_size=128, 
        filename='outputInsarStack.h5', access_mode='w', recon=False):
        """
        Initialize output arrays in H5 format.
        
        Parameters
        ----------
        data_shape: list or tuple
            Specifies the 3D shape of the interferogram array.
        tdec: ndarray
            Array of observation epochs in decimal year.
        statGen: {None, StationGenerator}, optional
            If provided, get lat/lon/elev from object.
        chunk_size: int, optional
            The size of the chunk chip for H5. Default: 128.
        filename: str, optional
            Output H5 filename. Default: 'outputInsarStack.h5'
        access_mode: str {'w', 'r'}, optional
            Access mode of the H5 arrays. Default: 'w' for write.
        recon: bool, optional
            The output data is reconstructed time series, instead of igram (False).
        """
        # Initialize variables common to all workers
        self.chunk_shape = [chunk_size, chunk_size]
        h5_chunk_shape = (1,chunk_size,chunk_size)
        self.recon = recon
        self.tdec = tdec
        self.Ny, self.Nx = data_shape[1:]
        self.nstat = self.Ny * self.Nx

        # Only master worker will have access to underlying data
        if self.rank == 0:

            # Open the H5 file and initialize data sets
            self.h5file = h5py.File(filename, access_mode)
            if self.recon:
                self._recon = self.h5file.create_dataset('recon', shape=data_shape, 
                    dtype=np.float32, chunks=h5_chunk_shape)
                self._data = self._recon
            else:
                self._igram = self.h5file.create_dataset('igram', shape=data_shape, 
                    dtype=np.float32, chunks=h5_chunk_shape)
                self._data = self._igram
            self._weights = self.h5file.create_dataset('weights', shape=data_shape, 
                dtype=np.float32, chunks=h5_chunk_shape)

            # Make a station generator
            if isinstance(statGen, StationGenerator):
                self.statGen = statGen
                self.statGen.los = self._data
                self.statGen.w_los = self._weights
                # Also make lat/lon/elev arrays in output H5
                for key in ['lat', 'lon', 'elev']:
                    self.h5file[key] = getattr(self.statGen, key)
            else:
                self.statGen = StationGenerator(self._igram, self._weights)

            ## And make a dictionary that just links to the generator
            self.statDict = self.statGen

            # Make sure we save tdec and 'insar' data type
            self.h5file['tdec'] = tdec
            self.h5file['dtype'] = 'insar'
            self.h5file['chunk_shape'] = self.chunk_shape

        # Barrier for safeguard
        self.comm.Barrier()
        return


    def initializeDataset(self, attr, N, chunk_size=None, access_mode='w'):
        """
        Create a new 3D array for saved H5 file.

        Parameters
        ----------
        attr: str
            String for H5 key.
        N: int
            Leading dimension for array (N,self.Ny,self.Nx).
        chunk_size: {None, int}, optional
            Chunk size. Defaults to saved chunk_size.
        """
        # Determine chunk shape
        if chunk_size is None:
            chunk_size = self.chunk_shape[0]
        h5_chunk_shape = (1,chunk_size,chunk_size)

        # Make new H5 Dataset and save
        if self.rank == 0:
            new_dset = self.h5file.create_dataset(attr, shape=(N,self.Ny,self.Nx),
                dtype=np.float32, chunks=h5_chunk_shape)
        else:
            new_dset = None
        setattr(self, '_%s' % attr, new_dset)

        return


    def getChunk(self, slice_y, slice_x, dtype='igram'):
        """
        Loads H5 data for a specified chunk given by slice objects.

        Parameters
        ----------
        slice_y: slice
            Slice of array in vertical dimension.
        slice_x: slice
            Slice of array in horizontal dimension.
        dtype: str {'igram', 'weight', 'par', 'recon'}, optional
            A string indicating which array to get the slice from:

            ``igram``
                The interferogram array. (Default)
            ``weight``
                The weight array (1 / sigma).
            ``par``
                The time series parameter array.
            ``recon``
                The reconstructed time series.

        Returns
        -------
        x: ndarray
            Array of data corresponding to specified chunk.
        """
        # Load data
        if self.rank == 0:
            arr = getattr(self, self.attr_dict[dtype])
            x = arr[:,slice_y,slice_x]
            x_shape = x.shape
        else:
            x_shape = None

        # Broadcast it
        x_shape = self.comm.bcast(x_shape, root=0)
        if self.rank != 0:
            x = np.empty(x_shape, dtype=np.float32)
        self.comm.Bcast([x, MPI.FLOAT], root=0)

        return x


    def setChunk(self, dat, slice_y, slice_x, dtype='igram', verbose=False):
        """
        Saves H5 data for a specified chunk given by slice objects.

        Parameters
        ----------
        dat: ndarray
            3D chunk array to save.
        slice_y: slice
            Slice of array in vertical dimension.
        slice_x: slice
            Slice of array in horizontal dimension.
        dtype: str {'igram', 'weight', 'par', 'recon'}, optional
            A string indicating which array to get the slice from:

            ``igram``
                The interferogram array. (Default)
            ``weight``
                The weight array (1 / sigma).
            ``par``
                The time series parameter array.
            ``recon``
                The reconstructed time series.
        verbose: bool, optional
            Print some statement. Default: False.
        """
        if self.rank == 0:
            if verbose: print('Saving chunk', (slice_y, slice_x))
            arr = getattr(self, self.attr_dict[dtype])
            arr[:,slice_y,slice_x] = dat
        return 


    def getMeanSlice(self, dtype='igram'):
        """
        Compute nanmean of 3D data along the time axis. See Insar.getChunk
        for supported 'dtype' strings.
        """
        mean_slice = None
        if self.rank == 0:
            data = getattr(self, self.attr_dict[dtype])
            mean_slice = np.nanmean(data, axis=0)
        return mean_slice 


    def getSlice(self, index, dtype='igram'):
        """
        Get LOS displacement for a given time index.

        Parameters
        ----------
        index: int
            Time index to extract slice.
        dtype: str, optional
            Data set to extract from. Default: 'igram'.

        Returns
        -------
        x: ndarray
            Extracted slice array.
        """
        kslice = None
        if self.rank == 0:
            data = getattr(self, self.attr_dict[dtype])
            kslice = np.array(data[index,:,:])
        return kslice


    def getPixel(self, row, col, dtype='igram'):
        """
        Get LOS displacement time series for a given pixel.

        Parameters
        ----------
        row: int
            Row index.
        col: int
            Column index.
        dtype: str, optional
            Data set to extract from. Default: 'igram'.

        Returns
        -------
        d: ndarray
            Extracted time series.
        """
        kts = None
        if self.rank == 0:
            data = getattr(self, self.attr_dict[dtype])
            kts = np.array(data[:,row,col])
        return kts


    def getData(self, dtype='igram'):
        """
        Get entire data array for a given dtype.

        Parameters
        ----------
        dtype: str, optional
            Data set to extract from. Default: 'igram'.
        """
        data_shape = data_type = None
        # Master loads the data
        if self.rank == 0:
            data = getattr(self, self.attr_dict[dtype])
            data = np.array(data)
            data_shape = data.shape
            data_type = data.dtype

        # Broadcast the parameters
        data_shape = self.comm.bcast(data_shape, root=0)
        data_type = self.comm.bcast(data_type, root=0)
        if self.rank != 0:
            data = np.empty(data_shape, dtype=data_type)

        # Use MPI to broadcast
        if data_type == np.float64:
            mpi_type = MPI.DOUBLE
        elif data_type == np.float32:
            mpi_type = MPI.FLOAT
        elif data_type == np.int32:
            mpi_type = MPI.INT
        else:
            assert False, 'Unhandled data type.'
        self.comm.Bcast([data, mpi_type], root=0)

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


def getChunks(data, chunk_y, chunk_x):
    """
    Utility function to get chunk bounds.

    Parameters
    ----------
    data: Insar
        Insar instance.
    chunk_y: int
        Size of chunk in vertical dimension.
    chunk_x: int
        Size of chunk in horizontal dimension.

    Returns
    -------
    chunks: list
        List of all chunks in the image.
    """
    # First determine the number of chunks in each dimension
    Ny_chunk = int(data.Ny // chunk_y)
    Nx_chunk = int(data.Nx // chunk_x)
    if data.Ny % chunk_y != 0:
        Ny_chunk += 1
    if data.Nx % chunk_x != 0:
        Nx_chunk += 1

    # Now construct chunk bounds
    chunks = []
    for i in range(Ny_chunk):
        if i == Ny_chunk - 1:
            nrows = data.Ny - chunk_y * i
        else:
            nrows = chunk_y
        istart = chunk_y * i
        iend = istart + nrows
        for j in range(Nx_chunk):
            if j == Nx_chunk - 1:
                ncols = data.Nx - chunk_x * j
            else:
                ncols = chunk_x
            jstart = chunk_x * j
            jend = jstart + ncols
            chunks.append([slice(istart,iend), slice(jstart,jend)])

    return chunks


# end of file
