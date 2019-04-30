#-*- coding: utf-8 -*-

import numpy as np
import tsinsar as ts
import shutil
import h5py

from scipy.spatial import cKDTree


class TimeSeries:
    """
    Abstract class for all time series data objects.
    """

    statDict = None

    def __init__(self, name="Time series", stnfile=None, dtype=None, copydict=False,
        h5file=None):

        self._name = name
        self.dtype = dtype

        # Determine the components depending on the datatype
        if self.dtype.lower() == 'gps':
            self.components = ['east', 'north', 'up']
        elif self.dtype.lower() in ['edm', 'insar']:
            self.components = ['los']
        elif self.dtype.lower() in ['wells']:
            self.components = ['up']
        else:
            assert False, 'Unsupported data type. Must be gps, edm, or insar'
        self.ncomp = len(self.components)

        self.h5file = self.output_h5file = self.statGen = None
        self.have_seasonal = False
        self.Jmat = None
        self.read_header = False

        # Read data if specified here
        if h5file is not None:
            self.loadStationH5(h5file, copydict=copydict)

        return


    def parse_line(self, line):
        """
        Child classes must define a line parser.
        """
        raise NotImplementedError('Child classes must define a line parser.')


    def parse_id(self, filepath):
        """
        Parse the file path to retrieve the station id, assuming a 4-character
        id. Can be overwritten by child classes for custom station ids.
        """
        return filepath.split('/')[-1][:4].lower()


    def setFormat(self, fmt):
        """
        Define the interface to the data dictionary.
        """
        if fmt == 'h5':
            self.getData = self._h5get
        else:
            self.getData = self._get
        return


    def getData(self, stat, attr):
        if type(stat) is dict:
            return np.array(stat[attr])
        else:
            return stat.attr


    def _h5get(self, attr):
        return np.array(self.h5file[attr])


    def _get(self, attr):
        return self.attr


    def __del__(self):
        """
        Make sure to close any H5 file.
        """
        if type(self.h5file) is h5py.File:
            print('Finished processing. Closing H5 file')
            self.h5file.close()
        elif type(self.h5file) is dict and self.output_h5file is not None:
            print('Finished processing. Saving H5 file')
            self._saveh5(self.output_h5file, self.h5file)
        return


    def clear(self):
        """
        Clears entries for station locations and names.
        """
        for attr in ('name', 'lat', 'lon', 'elev'):
            setattr(self, attr, [])
        self.nstat = 0
        return


    def loadStationH5(self, h5file, fileout=None, copydict=False):
        """
        Transfers data from an h5py data stack to self.
        """
        self.clear()

        # If user specifies an output file, we copy the input file and open
        # the output in read/write mode
        if fileout is None:
            if copydict:
                self.h5file = self._loadh5(h5file)
            else:
                self.h5file = h5py.File(h5file, 'r')
        else:
            shutil.copyfile(h5file, fileout)
            # Load all data into a python dictionary
            if copydict:
                self.h5file = self._loadh5(h5file)
                self.output_h5file = fileout
            else:
                self.h5file = h5py.File(fileout, 'r+')
        self.statDict = self.h5file

        # Make a generator to loop over station data
        self.makeStatGen()
        
        # Get the data
        self.transferDictInfo(h5=True)
        return


    def transferDictInfo(self, h5=True):
        """
        Transfer coordinates, names, and tdec from dictionary to self.
        """
        if not h5:
            raise NotImplementedError('still on todo list')
        for attr in ('name', 'lat', 'lon', 'elev'):
            setattr(self, attr, [])
        for statname, stat in self.statGen:
            self.name.append(statname)
            if type(stat['lat']) in [h5py.Dataset, h5py.Group]:
                self.lat.append(stat['lat'].value)
                self.lon.append(stat['lon'].value)
                self.elev.append(stat['elev'].value)
            else:
                self.lat.append(stat['lat'])
                self.lon.append(stat['lon'])
                self.elev.append(stat['elev'])
        self.name = np.array(self.name)
        self.lat,self.lon,self.elev = [np.array(lst) for lst in [self.lat,self.lon,self.elev]]
        self.nstat = self.lat.size
        self.tdec = np.array(self.h5file['tdec'])
        return

    
    def loadSeasonalH5(self, h5file):
        """
        Transfers data from an h5py data stack to self.
        """
        # Load the seasonal dictionary
        seas_dat = self._loadh5(h5file)

        # Get the number of periodic B-splines
        self.npbspline = seas_dat['npbspline']

        # Get the seasonal coefficients
        for statname, stat in self.statGen:
            seas_stat = seas_dat[statname]
            for component in self.components:
                m = seas_stat['m_' + component]
                stat['seasm_' + component] = m
                self.npar_seasonal = len(m)

        # Finally, make sure to remember that we have seasonal data
        self.have_seasonal = True
        return


    def read_metadata_ascii(self, filename, fmt, comment='#', delimeter=None):
        """
        Read coordinate metadata from an ASCII file.
        """
        self.clear()
        if filename is None:
            return

        # Parse the fmtdict string to make a dictionary
        columns = {}
        for keyval_str in fmt.split(','):
            key, value = keyval_str.split(':')
            columns[key.strip()] = int(value)

        with open(filename, 'r') as fid:
            for line in fid:
                if delimeter is None:
                    data = line.split()
                else:
                    data = line.split(delimeter)
                self.name.append(data[columns['id']])
                self.lat.append(data[columns['lat']])
                self.lon.append(data[columns['lon']])
                self.elev.append(data[columns['elev']])

        return


    def reformat_metadata(self, fmt='dict'):
        """
        Return metadata as {'dict', 'data frame'}.
        """
        out = {'id': self.name, 'lon': self.lon, 'lat': self.lat, 'elev': self.elev}
        if fmt == 'dict':
            return out
        elif fmt == 'data frame':
            import pandas as pd
            return pd.DataFrame(out)


    @staticmethod
    def _loadh5(h5file):
        """
        Load data from an H5 file into a dictionary.
        """
        data = {}
        with h5py.File(h5file, 'r') as h5file:
            for key, value in h5file.items():
                if type(value) == h5py.Group:
                    group = {}
                    for gkey, gvalue in value.items():
                        group[gkey] = gvalue.value
                    data[key] = group
                else:
                    data[key] = value.value
        return data
        

    @staticmethod
    def _saveh5(h5file, data):
        """
        Save a data stored in dictionary to H5 file.
        """
        print('H5 file:', h5file)
        with h5py.File(h5file, 'w') as hfid:
            for key, value in data.items():
                if type(value) is dict:
                    Group = hfid.create_group(key)
                    for gkey, gvalue in value.items():
                        Group[gkey] = gvalue
                else:
                    hfid[key] = value
        return


    def makeStatGen(self):
        """
        Make a generator for looping over the stations and station names.
        """
        self.statGen = list((statname, stat) for (statname, stat) in self.statDict.items()
            if statname not in ['tdec', 'G', 'cutoff', 'npbspline', 'dtype'])
        self.nstat = len(self.statGen)
        return


    def combine_data(self, filename):
        """
        Combine data from multiple stations. First chronological station is the reference one.
        """
        delete_stat = []
        with open(filename, 'r') as fid:
            # Loop over lines in file
            for line in fid:

                # Read the statnames
                statnames = [name.lower().strip() for name in line.split(',')]
                # Sort the stations by earliest finite data
                starting_indices = []
                for name in statnames:
                    data = self.statDict[name][self.components[0]]
                    starting_indices.append(np.isfinite(data).nonzero()[0][0])
                statnames = [statnames[ind] for ind in np.argsort(starting_indices)]
                ref_name = statnames[0]

                # Loop over the components
                for comp in self.components:
                    # Starting time representation is linear polynomial
                    rep = [['POLY',[1],[0.0]]]
                    # Get reference array to store everything
                    data = self.statDict[ref_name][comp]
                    wgt = self.statDict[ref_name]['w_' + comp]
                    for current_name in statnames[1:]:
                        # Get current array
                        current_data = self.statDict[current_name][comp]
                        current_wgt = self.statDict[current_name]['w_' + comp]
                        ind = np.isfinite(current_data).nonzero()[0]
                        # Store the data and weights
                        data[ind[0]:] = current_data[ind[0]:]
                        wgt[ind[0]:] = current_wgt[ind[0]:]
                        # Add heaviside
                        rep.append(['STEP',[self.trel[ind[0]]]])

                    # Remove ambiguities for any non-vertical components
                    if comp != 'up':
                        # Do least squares on finite data
                        ind = np.isfinite(data)
                        G = ts.Timefn(rep, self.trel)[0]
                        m = np.linalg.lstsq(G[ind,:], data[ind])[0]
                        # Remove only heaviside parts
                        data -= np.dot(G[:,2:], m[2:])

                # Remember the stations to delete
                delete_stat.extend(statnames[1:])
                
        # Delete redundant stations
        for name in delete_stat:
            del self.statDict[name]
                    
        # Some cleanup: remake the station generator
        self.makeStatGen() 
        self.transferDictInfo()
        return 


    def zeroMeanDisplacements(self):
        """
        Remove the mean of the finite values in each component of displacement.
        """
        from scipy.stats import nanmean
        for statname, stat in self.statGen:
            for component in self.components:
                dat = getattr(self, component)
                dat -= nanmean(dat)
        return


    def residuals(self):
        """
        Compute residuals between component and filt_component.
        """
        for comp in self.components:
            for statname, stat in self.statGen:
                data = stat[comp]
                filtered = stat['filt_' + comp]
                residual = data - filtered
                stat['residual_' + comp] = residual
        return 
 

    @staticmethod
    def adaptiveMedianFilt(dat, kernel_size):
        """
        Perform a median filter with a sliding window. For edges, we shrink window.
        """
        assert kernel_size % 2 == 1, 'kernel_size must be odd'

        nobs = dat.size
        filt_data = np.nan * np.ones_like(dat)

        # Beginning region
        halfWindow = 0
        for i in range(kernel_size//2):
            filt_data[i] = np.nanmedian(dat[i-halfWindow:i+halfWindow+1])
            halfWindow += 1

        # Middle region
        halfWindow = kernel_size // 2
        for i in range(halfWindow, nobs - halfWindow):
            filt_data[i] = np.nanmedian(dat[i-halfWindow:i+halfWindow+1])

        # Ending region
        halfWindow -= 1
        for i in range(nobs - halfWindow, nobs):
            filt_data[i] = np.nanmedian(dat[i-halfWindow:i+halfWindow+1])
            halfWindow -= 1

        return filt_data


    def computeStatDistance(self, statname1, statname2):
        """
        Compute the distance between two stations.
        """
        ind1 = (self.name == statname1.lower()).nonzero()[0]
        ind2 = (self.name == statname2.lower()).nonzero()[0]
        assert len(ind1) == 1, 'Cannot find first station'
        assert len(ind2) == 1, 'Cannot find second station'

        # Retrieve lat/lon
        lon1, lat1 = self.lon[ind1[0]], self.lat[ind1[0]]
        lon2, lat2 = self.lon[ind2[0]], self.lat[ind2[0]]

        # Convert to XYZ and compute Cartesian distance
        from topoutil import llh2xyz
        X1 = llh2xyz(lat1, lon1, 0.0, deg=True).squeeze()
        X2 = llh2xyz(lat2, lon2, 0.0, deg=True).squeeze()
        dX = np.linalg.norm(X2 - X1)
        return dX


    @property
    def tstart(self):
        return selt.tdec[0]
    @tstart.setter
    def tstart(self, val):
        raise AttributeError('Cannot set tstart explicitly')

    @property
    def trel(self):
        return self.tdec - self.tdec[0]
    @trel.setter
    def trel(self, val):
        raise AttributeError('Cannot set tstart explicitly')

    @property
    def numObs(self):
        return len(self.tdec)
    @numObs.setter
    def numObs(self, val):
        raise AttributeError('Cannot set numObs explicitly')



class GenericClass:
    pass
