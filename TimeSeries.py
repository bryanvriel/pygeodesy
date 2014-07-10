#-*- coding: utf-8 -*-

import numpy as np
import tsinsar as ts
import h5py


class TimeSeries:
    """
    Abstract class for all time series data objects.
    """

    def __init__(self, name="Time series", stnfile=None, dtype=None):

        self.name = name
        self.dtype = dtype

        # Read station/groundpoint locations from file       
        if stnfile is not None:
            ext = stnfile.split('.')[1]
            if ext == 'txt':
                name,lat,lon,elev = ts.tsio.textread(stnlist, 'S F F F')
            elif ext == 'h5':
                with h5py.File(stnfile, 'r') as ifid:
                    name = ifid['name'].value
                    lat = ifid['name'].value
                    lon = ifid['lon'].value
                    elev = ifid['elev'].value
            else:
                assert False, 'Input station file has an unsupported extension'

        # Determine the components depending on the datatype
        if self.dtype.lower() == 'gps':
            self.components = ['east', 'north', 'up']
        elif self.dtype.lower() in ['edm', 'insar']:
            self.components = ['los']
        else:
            assert False, 'Unsupported data type. Must be gps, edm, or insar'
        self.ncomp = len(self.components)

        return


    def clear(self):
        """
        Clears entries for station locations and names.
        """
        for attr in ('name', 'lat', 'lon', 'elev'):
            setattr(self, attr, [])
        self.nstat = 0
        return


    def loadStationDict(self, inputDict):
        """
        Transfers data from a dictionary of station classes to self.
        """
        self.clear()
        for statname, stat in inputDict.items():
            self.name.append(statname)
            self.lat.append(stat.lat)
            self.lon.append(stat.lon)
            self.elev.append(stat.elev)
        self.name = np.array(self.name)
        self.lat,self.lon,self.elev = [np.array(lst) for lst in [self.lat,self.lon,self.elev]]
        self.nstat = self.lat.size
        self.statDict = inputDict

        return


    def zeroMeanDisplacements(self):
        """
        Remove the mean of the finite values in each component of displacement.
        """
        from scipy.stats import nanmean
        for statname, stat in self.statDict.items():
            for component in self.components:
                dat = getattr(self, component)
                dat -= nanmean(dat)
        return
           

    def getDataArrays(self, order='columns'):
        """
        Traverses the station dictionary to construct regular sized arrays for the
        data and weights.
        """
        assert self.statDict is not None, 'must load station dictionary first'

        # Get first station to determine the number of data points
        for statname, stat in self.statDict.items():
            dat = getattr(stat, self.components[0])
            ndat = dat.size
            break 

        # Construct regular arrays and fill them
        data = np.empty((ndat, self.nstat*self.ncomp))
        weights = np.empty((ndat, self.nstat*self.ncomp))
        j = 0
        for component in self.components:
            for statname, cnt in zip(self.name, range(self.nstat)):
                stat = self.statDict[statname]
                compDat = getattr(stat, component)
                compWeight = getattr(stat, 'w_' + component)
                data[:,j] = compDat
                weights[:,j] = compWeight
                j += 1
        
        if order == 'rows':
            return data.T.copy(), weights.T.copy()
        else:
            return data, weights


    def computeNetworkWeighting(self, smooth=1.0, n_neighbor=3, L0=None):
        """
        Computes the network-dependent spatial weighting based on station/ground locations.
        """
        import topoutil as tu

        dist_weight = np.zeros((self.nstat, self.nstat))
        # Loop over stations
        rad = np.pi / 180.0
        for i in range(self.nstat):
            ref_X = tu.llh2xyz(self.lat[i]*rad, self.lon[i]*rad, self.elev[i])
            stat_dist = np.zeros((self.nstat,))
            # Loop over other stations
            for j in range(self.nstat):
                if j == i:
                    continue
                curr_X = tu.llh2xyz(self.lat[j]*rad, self.lon[j]*rad, self.elev[j])
                # Compute distance between stations
                stat_dist[j] = np.linalg.norm(ref_X - curr_X)
            if L0 is None:
                # Mean distance to 3 nearest neighbors multipled by a smoothing factor
                Lc = smooth * np.mean(np.sort(stat_dist)[1:1+n_neighbor])
                dist_weight[i,:] = np.exp(-stat_dist / Lc)
                print(' - scale length at', self.name[i], ':', 0.001 * Lc, 'km')
            else:
                dist_weight[i,:] = np.exp(-stat_dist / L0)

        return dist_weight


class GenericClass:
    pass
