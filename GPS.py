#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from .Station import STN
from scipy.spatial import cKDTree
import SparseConeQP as sp
from matutils import dmultl, dmultr
from timeutils import generateRegularTimeArray
import tsinsar as ts
import sys, os

from .TimeSeries import TimeSeries


class GPS(TimeSeries):
    """
    Class to hold GPS station data and perform inversions
    """
    
    statDict = None

    def __init__(self, name='gps', stnfile=None, datformat=None):
        """
        Initiate GPS structure with station list
        """
        # Initialize the parent class
        super().__init__(name=name, stnfile=stnfile, dtype='gps')

        self.datformat = datformat
        return

    
    def read_data(self, gpsdir, fileKernel='CleanFlt', dataFactor=1000.0):
        """
        Reads GPS data and convert to mm
        """
        self.stns = []
        lon = []; lat = []; elev = []; name = []
        for ii in range(self.nstat):
            stn = STN(self.name[ii], gpsdir, format=self.datformat,
                fileKernel=fileKernel, dataFactor=dataFactor)
            if stn.success:
                self.stns.append(stn)
                lon.append(self.lon[ii])
                lat.append(self.lat[ii])
                elev.append(self.elev[ii])
                name.append(self.name[ii])
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


    def resample(self, interp=False, t0=None, tf=None):
        """
        Resample GPS data to a common time array
        """

        # Loop through stations to find bounding times
        tmin = 1000.0
        tmax = 3000.0
        for stn in self.stns:
            tmin_cur = stn.tdec.min()
            tmax_cur = stn.tdec.max()
            if tmin_cur > tmin:
                tmin = tmin_cur
            if tmax_cur < tmax:
                tmax = tmax_cur

        refflag = False
        if t0 is not None and tf is not None:
            refflag = True
            tref = generateRegularTimeArray(t0, tf)
            days = tref.size
            tree = cKDTree(tref.reshape((days,1)), leafsize=2*days)
        else:
            tref = generateRegularTimeArray(tmin, tmax)
            days = tref.size

        # Retrieve data that lies within the common window
        for stn in self.stns:
            if interp:
                stn.north = np.interp(tref, stn.tdec, stn.north)
                stn.east = np.interp(tref, stn.tdec, stn.east)
                stn.up = np.interp(tref, stn.tdec, stn.up)
                stn.tdec = tref.copy()
            elif t0 is not None and tf is not None:
                north = np.nan * np.ones_like(tref)
                east = np.nan * np.ones_like(tref)
                up = np.nan * np.ones_like(tref)
                dn = np.nan * np.ones_like(tref)
                de = np.nan * np.ones_like(tref)
                du = np.nan * np.ones_like(tref)
                for i in range(stn.tdec.size):
                    if stn.tdec[i] < t0 or stn.tdec[i] > tf:
                        continue
                    nndist, ind = tree.query(np.array([stn.tdec[i]]), k=1, eps=1.0)
                    north[ind] = stn.north[i]
                    east[ind] = stn.east[i]
                    up[ind] = stn.up[i]
                    dn[ind] = stn.sn[i]
                    de[ind] = stn.se[i]
                    du[ind] = stn.su[i]
                stn.tdec, stn.north, stn.east, stn.up = tref, north, east, up
                stn.sn, stn.se, stn.su = dn, de, du
            else:
                bool = (stn.tdec >= tmin) & (stn.tdec <= tmax)
                stn.north = stn.north[bool]
                stn.east = stn.east[bool]
                stn.up = stn.up[bool]
                stn.tdec = stn.tdec[bool]


    def extended_spinvert(self, tdec, repDict, penalty, cutoffDict, maxiter=4,
                          outlierThresh=1.0e6):
        """
        Performs sparse inversion of model coefficients on each component. Each
        station will have its own time representation and its own cutoff.
        """
        # Loop over the stations
        mDicts = {}
        for statname, stat in self.statGen:

            # Construct a G matrix
            Gref = np.asarray(ts.Timefn(repDict[statname], tdec-tdec[0])[0], order='C')
            ndat,Npar = Gref.shape
            refCutoff = cutoffDict[statname]

            # Loop over the components
            mDict = {}
            for comp, w_comp in [('east','w_e'), ('north','w_n'), ('up','w_u')]:

                #if comp in ['east', 'north']:
                #    G = Gref[:,4:]
                #    cutoff = refCutoff - 4
                #else:
                #    G = Gref
                #    cutoff = refCutoff
                cutoff = refCutoff
                G = Gref

                # Get finite data
                dat = (getattr(stat, comp)).copy()
                ind = np.isfinite(dat)
                dat = dat[ind]
                wgt = getattr(stat, w_comp)[ind]

                # Instantiate a solver
                solver = sp.BaseOpt(cutoff=cutoff, maxiter=maxiter, weightingMethod='log')

                # Perform estimation
                m = solver.invert(dmultl(wgt, G[ind,:]), wgt*dat, penalty)[0]

                # Do one pass to remove outliers
                fit = np.dot(G, m)
                raw_dat = getattr(stat, comp)
                misfit = np.abs(raw_dat - fit)
                ind = misfit > outlierThresh
                if ind.nonzero()[0].size > 1:
                    print('Doing another pass to remove outliers')
                    raw_dat[ind] = np.nan
                    finiteInd = np.isfinite(raw_dat)
                    dat = raw_dat[finiteInd]
                    wgt = getattr(stat, w_comp)[finiteInd]
                    m = solver.invert(dmultl(wgt, G[finiteInd,:]), wgt*dat, penalty)[0]

                #if comp in ['east', 'north']:
                #    m = np.hstack((np.zeros((4,)), m))
                mDict[comp] = m
            mDicts[statname] = (mDict, G)

        return mDicts

# end of file
