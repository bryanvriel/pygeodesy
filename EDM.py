
import numpy as np
import matplotlib.pyplot as plt
import tsinsar as ts
import h5py
import sys
import os

from .TimeSeries import TimeSeries

class EDM(TimeSeries):
    """
    Class to hold GPS station data and perform inversions
    """

    def __init__(self, name='edm', stnfile=None, stnlist=None):
        """
        Initiate EDM class.
        """
        super().__init__(name=name, stnfile=stnfile, dtype='edm')
        return
        

    def compute_los2casa(self):
        """
        Computes the NEU look vector from CASA to each station
         - currently a little klunky, but works
        """

        rad = np.pi / 180.0

        # CASA coordinates
        clat = np.array([37.64434433 * rad])
        clon = np.array([-118.89565277 * rad])
        celev = np.array([2410.0])
        casa_pos = tpu.llh2xyz(clat, clon, celev)

        # Loop through stations
        self.dr_casa = []
        for ii in range(self.nstat):
            lat = np.array([self.lat[ii]*rad])
            lon = np.array([self.lon[ii]*rad])
            elev = np.array([self.elev[ii]])
            stat_pos = tpu.llh2xyz(lat, lon, elev)
            dr = np.reshape(stat_pos - casa_pos, (3,1))
            enu = tpu.geo2topo(dr, casa_pos)[0]
            enu /= np.linalg.norm(enu)
            neu = [enu[1], enu[0], enu[2]]
            self.dr_casa.append(neu)

        return


# end of file
