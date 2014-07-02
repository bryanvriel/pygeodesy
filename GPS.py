

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from .Station import STN
from .MPClasses import MPInvert, MPWeights, makeSharedArrays
from mpi4py import MPI
from scipy.spatial import cKDTree
from scipy.stats import nanmean
import SparseConeQP as sp
#from tnipm import TNIPM
import topoutil as tu
from matutils import dmultl, dmultr
import tsinsar as ts
#from .sopac import sopac
import sys, os


class GPS:
    """
    Class to hold GPS station data and perform inversions
    """
    
    statDict = None

    def __init__(self, stnlist, format='sopac', fileKernel='CleanFlt'):
        """
        Initiate GPS structure with station list
        """

        [name,lat,lon,elev] = ts.tsio.textread(stnlist, 'S F F F')
        self.name = np.array(name)
        self.lat = np.array(lat)
        self.lon = np.array(lon)
        self.elev = np.array(elev)
        self.nstat = len(self.name)
        self.format = format
        self.fileKernel = fileKernel
        assert format in ['sopac', 'pbo', 'geonetnz'], 'unhandled format'

    def clear(self):
        """
        Clears entries for station locations and names.
        """
        self.name = []
        self.lat = []
        self.lon = []
        self.elev = []
        self.nstat = 0

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
        self.lat,self.lon,self.elev = [np.array(list) for list in [self.lat,self.lon,self.elev]]
        self.nstat = self.lat.size
        self.statDict = inputDict

        return


    def zeroMeanDisplacements(self):
        """
        Remove the mean of the finite values in each component.
        """
        for statname, stat in self.statDict.items():
            stat.north -= nanmean(stat.north)
            stat.east -= nanmean(stat.east)
            stat.up -= nanmean(stat.up)

        return


    def read_data(self, gpsdir, dataFactor=1000.0):
        """
        Reads GPS data and convert to mm
        """
        self.stns = []
        for ii in range(self.nstat):
            self.stns.append(STN(self.name[ii], gpsdir, format=self.format,
                                 fileKernel=self.fileKernel, dataFactor=dataFactor))

    def preprocess(self, hdrformat='filt', dataFactor=1000.0):
        """
        Preprocess the data to remove any known offsets as listed in the 
        Sopac header file.
        """

        # Don't do anything if we're not using Sopac
        if 'sopac' not in self.format:
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


    def getDataArrays(self, order='columns'):
        """
        Traverses the station dictionary to construct regular sized arrays for the
        data and weights.
        """
        assert self.statDict is not None, 'must load station dictionary first'

        # Get first station to determine the number of data points
        for statname, stat in self.statDict.items():
            ndat = stat.east.size
            break

        # Construct regular arrays and fill them
        north = np.empty((ndat, self.nstat))
        east = np.empty((ndat, self.nstat))
        up = np.empty((ndat, self.nstat))
        w_north = np.empty((ndat, self.nstat))
        w_east = np.empty((ndat, self.nstat))
        w_up = np.empty((ndat, self.nstat))
        for statname, cnt in zip(self.name, range(self.nstat)):
            stat = self.statDict[statname]
            north[:,cnt], east[:,cnt], up[:,cnt] = stat.north, stat.east, stat.up
            w_north[:,cnt], w_east[:,cnt], w_up[:,cnt] = stat.w_n, stat.w_e, stat.w_u

        if order == 'rows':
            return (east.T.copy(), north.T.copy(), up.T.copy(), 
                    w_east.T.copy(), w_north.T.copy(), w_up.T.copy())
        else:
            return east, north, up, w_east, w_north, w_up


    def resample(self, interp=False, t0=None, tf=None):
        """
        Resample GPS data to a common time array
        """

        # Loop through stations to find bounding times
        tmin = 1000.0
        tmax = 3000.0
        for stn in self.stns:
            tmin_cur = stn.tdec[0]
            tmax_cur = stn.tdec[-1]
            if tmin_cur > tmin:
                tmin = tmin_cur
            if tmax_cur < tmax:
                tmax = tmax_cur

        tcommon = np.linspace(tmin, tmax, int((tmax-tmin)*365.0))
        refflag = False
        if t0 is not None and tf is not None:
            refflag = True
            days = int((tf - t0) * 365.0)
            tref = np.linspace(t0, tf, days)
            tree = cKDTree(tref.reshape((days,1)), leafsize=2*days)

        # Retrieve data that lies within the common window
        for stn in self.stns:
            if interp:
                stn.north = np.interp(tcommon, stn.tdec, stn.north)
                stn.east = np.interp(tcommon, stn.tdec, stn.east)
                stn.up = np.interp(tcommon, stn.tdec, stn.up)
                stn.tdec = tcommon.copy()
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
        for statname, stat in self.statDict.items():

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


    def spinvert(self, tdec, rep, penalty, cutoff, maxiter=4, nproc=1):
        """
        Performs sparse inversion of model coefficients on each component
        """

        # Construct a reference G to get the number of parameters
        Gref = np.asarray(ts.Timefn(rep, tdec-tdec[0])[0], order='C')
        ndat,Npar = Gref.shape
       
        # Find indices of valid data for each station 
        boolList = []
        north = np.empty((ndat, self.nstat))
        east = np.empty((ndat, self.nstat))
        up = np.empty((ndat, self.nstat))
        w_n = np.empty((ndat, self.nstat))
        w_e = np.empty((ndat, self.nstat))
        w_u = np.empty((ndat, self.nstat))
        cnt = 0
        for statname in self.statDict:
            stat = self.statDict[statname]
            boolList.append((np.isfinite(stat.east), 
                             np.isfinite(stat.north),
                             np.isfinite(stat.up)))
            north[:,cnt], east[:,cnt], up[:,cnt] = stat.north, stat.east, stat.up
            w_n[:,cnt], w_e[:,cnt], w_u[:,cnt] = stat.w_n, stat.w_e, stat.w_u
            cnt += 1
         
        # Make shared memory Arrays with certain shapes
        shared = GenericClass()
        pshape = (self.nstat, Npar-cutoff)
        mshape = (Npar, self.nstat)
        mpArrays = makeSharedArrays([pshape, pshape, pshape, mshape, mshape, mshape])
        shared.penn, shared.pene, shared.penu = mpArrays[:3]
        shared.m_north, shared.m_east, shared.m_up = mpArrays[3:]

        # Assign uniform penalty for all components
        penn, pene, penu = penalty, penalty, penalty

        # Partition data over several processors for inversions
        nominal_load = self.nstat // nproc
        istart = 0
        threads = []
        for id in range(nproc):
            # Get number of stations to process for this worker
            if id == nproc - 1:
                procN = self.nstat - id * nominal_load
            else:
                procN = nominal_load
            # Subset the data
            ind = range(istart, istart+procN)
            subbed = [[boolList[index] for index in ind], 
                      north[:,ind], east[:,ind], up[:,ind], 
                      w_n[:,ind], w_e[:,ind], w_u[:,ind],
                      penn, pene, penu]

            # Initialize solver
            l1 = sp.BaseOpt(cutoff=cutoff, maxiter=maxiter, weightingMethod='log')
           
            # Send to a processor 
            threads.append(MPInvert(shared, subbed, Gref, l1, istart, cutoff))
            threads[id].start()
            istart += procN

        # Wait for everyone to finish
        for thread in threads:
            thread.join()

        return shared.m_north, shared.m_east, shared.m_up, Gref


    def group_invert(self, tdec, north, east, up, w_n, w_e, w_u, rep, penalty, cutoff, maxiter=4):
        """
        Invert all stations as a group. Perform one iteration, and then compute
        the average penalty vector for all stations. Use this average weight as
        an input for the next iteration. Note that all stations must be sampled
        to the same time array in order for the weights to make sense from station
        to station.
        """

        # First, construct a reference G to get the number of parameters
        Gref = np.asarray(ts.Timefn(rep, tdec-tdec[0])[0], order='C')
        N,Npar = Gref.shape

        # Initial penalty arrays
        penn = penalty * np.ones((Npar-cutoff,), dtype=float)
        pene = penalty * np.ones((Npar-cutoff,), dtype=float)
        penu = penalty * np.ones((Npar-cutoff,), dtype=float)

        # Find indices of valid data for each station
        bool_list = []
        for i in range(self.nstat):
            bool_list.append(np.isfinite(north[:,i]))
        
        # Initialize solver
        l1 = sp.BaseOpt(cutoff=cutoff, maxiter=1, weightingMethod='log')

        # Begin iterations
        print('\nPerforming group inversion')
        m_north = np.zeros((Npar,self.nstat))
        m_east = np.zeros((Npar,self.nstat))
        m_up = np.zeros((Npar,self.nstat))
        pen_old = np.zeros((Npar-cutoff,))
        factor = 1
        for ii in range(maxiter):
            print(' - at iteration', ii)
            #allpen = 0.0
            allpen = np.zeros((self.nstat, Npar-cutoff))
            for jj in range(self.nstat):
                # Get indices of valid observations
                bool = bool_list[jj]
                G = Gref[bool,:]
                dnorth, deast, dup = north[bool,jj], east[bool,jj], up[bool,jj]
                wn, we, wu = w_n[bool,jj], w_e[bool,jj], w_u[bool,jj]
                # Perform estimation and store weights
                m_north[:,jj], qn = l1.invert(dmultl(wn, G), wn*dnorth, penn)
                m_east[:,jj], qe  = l1.invert(dmultl(we, G), we*deast, pene)
                m_up[:,jj], qu    = l1.invert(dmultl(wu, G), wu*dup, penu)
                #allpen += qn[cutoff:] + qe[cutoff:] 
                allpen[jj,:] = qn[cutoff:] + qe[cutoff:]

            # Compute average weight
            #avgpen = allpen / (2.0 * self.nstat)
            avgpen = np.median(allpen, axis=0)
            penn = penalty * avgpen
            pene = penalty * avgpen
            penu = penalty * avgpen

            pdiff = avgpen - pen_old
            print(' - weight difference:', np.std(pdiff))
            pen_old = avgpen.copy()

            plt.semilogy(factor*pene, label='%d'%ii)
            factor *= 10.0

        plt.legend()
        plt.show()

        return m_north, m_east, m_up, Gref


    def weighted_group_invert(self, tdec, rep, penalty, cutoff, maxiter=1, power=1, L0=None,
                              wtype='mean', weightingMethod='inverse', smooth=1.0,
                              viewStations=[], zeroMean=False, G=None, solver='cvxopt',
                              communicator=None, norm_tol=1.0e-4):
        """
        Invert all stations as a group. Perform one iteration, and then compute
        the average penalty vector for all stations. Use this average weight as
        an input for the next iteration. Note that all stations must be sampled
        to the same time array in order for the weights to make sense from station
        to station.
        """

        # Make sure I have a valid communicator
        self.comm = communicator or MPI.COMM_WORLD
        # Get the number of processors
        self.tasks = self.comm.Get_size()
        # And my rank
        self.rank = self.comm.Get_rank()

        # Determine my workload
        nominal_load = self.nstat // self.tasks
        if self.rank == self.tasks - 1:
            procN = self.nstat - self.rank * nominal_load
        else:
            procN = nominal_load

        # Choose my estimator for the weights
        if 'mean' == wtype:
            estimator = MPWeights.computeMean
        elif 'median' == wtype:
            estimator = MPWeights.computeMedian
        else:
            assert False, 'unsupported weight type. must be mean or median'

        # If I am the master, do some prep work
        if self.rank == 0:

            # Pre-compute distances between stations to all other stations and 
            # corresponding weights
            rad = np.pi / 180.0
            dist_weight = self.computeNetworkWeighting(smooth=smooth, L0=L0)

            # Construct a reference G to get the number of parameters if G is not provided
            if G is None:
                Gref = np.asarray(ts.Timefn(rep, tdec-tdec[0])[0], order='C')
            else:
                Gref = G.copy()
            N,Npar = Gref.shape
            if zeroMean:
                for j in range(Npar):
                    Gref -= np.mean(Gref[:,j])

            # Get regular data arrays; row contiguous
            east0, north0, up0, w_east0, w_north0, w_up0 = self.getDataArrays(order='rows')
            print('Data shape:', east0.shape)

            # Specify the solver type
            if solver == 'cvxopt':
                solverClass = sp.BaseOpt
            elif solver == 'tnipm':
                try:
                    from tnipm import TNIPM
                    solverClass = TNIPM
                except ImportError:
                    solverClass = sp.BaseOpt
            else:
                assert False, 'unsupported solver type'

            # Instantiate a solver
            objSolver = solverClass(cutoff=cutoff, maxiter=1, eps=1.0e-2,
                                    weightingMethod=weightingMethod)

            # Allocate arrays to store the final results
            mShape = (self.nstat, Npar)
            m_east0 = np.zeros(mShape)
            m_north0 = np.zeros(mShape)
            m_up0 = np.zeros(mShape)
            pene0 = np.zeros(mShape)
            penn0 = np.zeros(mShape)
            penu0 = np.zeros(mShape)

        else:
            # Workers know nothing about the solver or design matrix
            objSolver = Gref = None
            # Nor do they know about the data
            east0 = north0 = up0 = w_east0 = w_north0 = w_up0 = None
            # Nor about the final results
            m_east0 = m_north0 = m_up0 = pene0 = penn0 = penu0 = None

        # Broadcast the solver and design matrix
        objSolver = self.comm.bcast(objSolver, root=0)
        Gref = self.comm.bcast(Gref, root=0)
        Npar = Gref.shape[1]

        # Allocate sub arrays to hold my part of the data
        subShape = (procN, Gref.shape[0])
        east = np.empty(subShape)
        north = np.empty(subShape)
        up = np.empty(subShape)
        w_east = np.empty(subShape)
        w_north = np.empty(subShape)
        w_up = np.empty(subShape)

        # Create row data types for data and parameter chunks
        dat_rowtype = MPI.DOUBLE.Create_contiguous(Gref.shape[0])
        dat_rowtype.Commit()
        par_rowtype = MPI.DOUBLE.Create_contiguous(Gref.shape[1])
        par_rowtype.Commit()

        # Scatter
        sendcnts = self.comm.gather(procN, root=0)
        self.comm.Scatterv([east0,  (sendcnts, None), dat_rowtype], east, root=0)
        self.comm.Scatterv([north0, (sendcnts, None), dat_rowtype], north, root=0)
        self.comm.Scatterv([up0,    (sendcnts, None), dat_rowtype], up, root=0)
        self.comm.Scatterv([w_east0,  (sendcnts, None), dat_rowtype], w_east, root=0)
        self.comm.Scatterv([w_north0, (sendcnts, None), dat_rowtype], w_north, root=0)
        self.comm.Scatterv([w_up0,    (sendcnts, None), dat_rowtype], w_up, root=0)

        # Allocate arrays for model coefficients and penalty weights
        assert type(penalty) is dict, 'this routine only handles dictionary of penalties'
        mShape = (procN, Npar)
        m_east = np.zeros(mShape)
        m_north = np.zeros(mShape)
        m_up = np.zeros(mShape)
        pene = penalty['north'] * np.ones(mShape)
        penn = penalty['east'] * np.ones(mShape)
        penu = penalty['up'] * np.ones(mShape)

        # Iterate
        for ii in range(maxiter):

            if self.rank == 0: print('At iteration', ii)

            # Loop over my portion of GPS stations
            for jj in range(procN):

                # Get boolean indices of valid observations and weights
                ind = (np.isfinite(east[jj,:]) * np.isfinite(north[jj,:]) 
                     * np.isfinite(up[jj,:]) * np.isfinite(w_east[jj,:])
                     * np.isfinite(w_north[jj,:]) * np.isfinite(w_up[jj,:]))
                Gsub = Gref[ind,:]

                # Get my penalties
                eastPen = pene[jj,cutoff:]
                northPen = penn[jj,cutoff:]
                upPen = penu[jj,cutoff:]

                # Perform estimation and store weights
                m_north[jj,:], qn = objSolver.invert(dmultl(w_north[jj,ind],Gsub), 
                                                     w_north[jj,ind] * north[jj,ind], 
                                                     northPen)
                m_east[jj,:], qe = objSolver.invert(dmultl(w_east[jj,ind],Gsub),
                                                    w_east[jj,ind] * east[jj,ind], 
                                                    eastPen)
                m_up[jj,:], qu = objSolver.invert(dmultl(w_up[jj,ind],Gsub),
                                                  w_up[jj,ind] * up[jj,ind], 
                                                  upPen)
    
                # Now modify the penalty array
                penn[jj,:] = qn
                pene[jj,:] = qe
                penu[jj,:] = qu

            self.comm.Barrier()

            # Master gathers the results for this iteration
            self.comm.Gatherv(m_east,  [m_east0, (sendcnts, None), par_rowtype], root=0)
            self.comm.Gatherv(m_north, [m_north0, (sendcnts, None), par_rowtype], root=0)
            self.comm.Gatherv(m_up,    [m_up0, (sendcnts, None), par_rowtype], root=0)
            self.comm.Gatherv(pene, [pene0, (sendcnts, None), par_rowtype], root=0)
            self.comm.Gatherv(penn, [penn0, (sendcnts, None), par_rowtype], root=0)
            self.comm.Gatherv(penu, [penu0, (sendcnts, None), par_rowtype], root=0)

            # Master computes the weights
            if self.rank == 0:

                # Temporaries
                pene_new = np.zeros_like(pene0)
                penn_new = np.zeros_like(penn0)
                penu_new = np.zeros_like(penu0)
    
                for jj in range(self.nstat):
                    # Distance weights
                    weights = np.tile(np.expand_dims(dist_weight[jj,:], axis=1), (1,Npar))
                    # Compute weighted penalty
                    penn_new[jj,:] = estimator(weights, penn0, penalty['north'])
                    pene_new[jj,:] = estimator(weights, pene0, penalty['east'])
                    penu_new[jj,:] = estimator(weights, penu0, penalty['up'])

                # Check the exit condition
                wDiff = penn_new - penn0
                normDiff = np.linalg.norm(wDiff, ord='fro')
                print(' - normDiff =', normDiff)
                if normDiff < norm_tol:
                    exitFlag = True
                else:
                    exitFlag = False

                # Copy from temporaries
                pene0[:,:] = pene_new
                penn0[:,:] = penn_new
                penu0[:,:] = penu_new

            else:
                exitFlag = None

            # Scatter the updated penalties
            self.comm.Scatterv([pene0, (sendcnts, None), par_rowtype], pene, root=0)
            self.comm.Scatterv([penn0, (sendcnts, None), par_rowtype], penn, root=0)
            self.comm.Scatterv([penu0, (sendcnts, None), par_rowtype], penu, root=0)

            # Broadcast the exit flag
            exitFlag = self.comm.bcast(exitFlag, root=0)
            if exitFlag:
                break

        dat_rowtype.Free()
        par_rowtype.Free()

        # Done
        if self.rank == 0:
            return m_north0.T, m_east0.T, m_up0.T, Gref
        else:
            return None, None, None, None

    
    def computeNetworkWeighting(self, smooth=1.0, n_neighbor=3, L0=None):
        """
        Computes the network-dependent spatial weighting.
        """
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


    def viewIteration(self, statname, shared, east_pen, north_pen, G, tdec, itcnt):

        fig = plt.figure(figsize=(16,8))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        axes_pairs = [(ax1, ax3), (ax2, ax4)]

        # Compute current transient signal
        statind = (self.name == statname).nonzero()[0]
        cnt = 0
        for component, penarray in [('east', east_pen), ('north', north_pen)]:

            m = getattr(shared, 'm_' + component)[:,statind].squeeze()
            pen = penarray[statind,:].squeeze()
            npar = pen.size
            fitc = np.dot(G[:,:self.cutoff], m[:self.cutoff])
            fitt = np.dot(G[:,self.cutoff:], m[self.cutoff:])
            dat = getattr(self.statDict[statname], component)

            ax_top, ax_bottom = axes_pairs[cnt]
            
            ax_top.plot(tdec, dat - fitc, 'o', alpha=0.6)
            ax_top.plot(tdec, fitt, '-r', linewidth=3)
            ax_top.set_xlim([tdec[0], tdec[-1]])
            ax_bottom.plot(np.arange(npar), pen)
            ax_bottom.set_xlim([0, npar])
            cnt += 1

        plt.suptitle(statname, y=0.94, fontsize=18)
        #plt.show(); assert False
        fig.savefig('figures/%s_iterations_%03d.png' % (statname, itcnt), bbox_inches='tight')
        plt.close(fig)
        return 


    def xval(self, kfolds, lamvec, rep, cutoff, tdec, maxiter=1, statlist=None, plot=False):
        """
        Performs k-fold cross-validation on GPS data
        """
        if statlist is None:
            statlist = self.name

        # Make representaton and cutoff dictionaries if not provided
        if type(rep) is not dict:
            repDict = {}
            for statname in statlist:
                repDict[statname] = rep
        else:
            repDict = rep

        if type(cutoff) is not dict:
            cutoffDict = {}
            for statname in statlist:
                cutoffDict[statname] = cutoff
        else:
            cutoffDict = cutoff
        
        if not os.path.exists('figures'):
            os.mkdir('figures')

        for statname in statlist:
            
            stat = self.statDict[statname]
            print(' - cross validating at station', statname)

            #plt.plot(tdec, stat.north, 'o'); plt.title(statname); plt.show(); assert False

            # Construct dictionary matrix
            Gref = np.asarray(ts.Timefn(repDict[statname], tdec-tdec[0])[0], order='C')
            N,Npar = Gref.shape
            cutoff = cutoffDict[statname]

            # Find indices for valid data
            ind = np.isfinite(stat.north)
            G = Gref[ind,:].copy()
            dnorth, deast, dup = stat.north[ind], stat.east[ind], stat.up[ind]
            wn, we, wu = stat.w_n[ind], stat.w_e[ind], stat.w_u[ind]

            # Sparse Cone QP solver
            l1 = sp.BaseOpt(cutoff=cutoff, maxiter=maxiter)
            pen_n, err_north = l1.xval(kfolds, lamvec, dmultl(wn, G), wn*dnorth)
            print('    - finished north with optimal penalty of', pen_n)
            pen_e, err_east = l1.xval(kfolds, lamvec, dmultl(we, G), we*deast)
            print('    - finished east with optimal penalty of', pen_e)
            pen_u, err_up = l1.xval(kfolds, lamvec, G, dup)
            print('    - finished up with optimal penalty of', pen_u)

            if plot:
                fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
                ax1.semilogx(lamvec, err_east)
                ax2.semilogx(lamvec, err_north)
                ax3.semilogx(lamvec, err_up)
                fig.savefig('figures/xval_%s.png' % (statname))
                plt.clf()

                
class GenericClass:
    pass

