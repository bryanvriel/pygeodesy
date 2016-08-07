#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import tsinsar as ts
from .Solver import Solver
from .MPClasses import MPWeights
import sys

dmultl = ts.dmultl

class MPISolver(Solver):
    """
    Use mpi4py to do inversions in parallel.
    """

    def __init__(self, data, timeRep, solver, penaltyDict, communicator=None, 
                 smooth=1.0, L0=None, seasrep=None):
        """
        Initialize the parent Solver class and get my MPI properties.
        """
        super().__init__(data, timeRep, solver, seasrep=seasrep)
        from mpi4py import MPI
        self.comm = communicator or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.tasks = self.comm.Get_size()
        self.m_seas0 = None

        # Extra initialization for master processor
        if self.rank == 0:

            # Get data and weight matrices; row contiguous
            if self.data.have_seasonal:
                self.datArr0, self.wgtArr0, self.m_seas0 = self.data.getDataArrays(order='rows')
            else:
                self.datArr0, self.wgtArr0 = self.data.getDataArrays(order='rows')
            assert self.ndat == self.datArr0.shape[1] 

            # Compute distance weights
            self.dist_weight = data.computeNetworkWeighting(smooth=smooth, L0=L0)

            # Allocate arrays to store final results from all workers
            mShape = (self.nstat*self.ncomp, self.npar)
            self.m0 = np.zeros(mShape)
            self.λ0 = np.ones(mShape)

            # Make 1-d array of penalties to scatter among workers
            assert type(penaltyDict) is dict
            self.penalties0 = np.ones((self.nstat*self.ncomp,))
            ind_beg = 0
            for component in data.components:
                ind_end = ind_beg + self.nstat
                self.penalties0[ind_beg:ind_end] *= penaltyDict[component]
                ind_beg += self.nstat

        else:
            self.datArr0 = self.wgtArr0 = None
            self.m0 = self.λ0 = None
            self.penalties0 = None
            
        return


    def partitionData(self):
        """
        Scatter the data among the workers.
        """
        from mpi4py import MPI
        N = self.nstat * self.ncomp
        nominal_load = N // self.tasks
        if self.rank == self.tasks - 1:
            self.procN = N - self.rank * nominal_load
        else:
            self.procN = nominal_load

        # Allocate sub arrays to hold my part of the data
        subShape = (self.procN, self.ndat)
        self.datArr = np.empty(subShape)
        self.wgtArr = np.empty(subShape)        
               
        # Create row data types for data and parameter chunks
        self.dat_rowtype = MPI.DOUBLE.Create_contiguous(self.ndat)
        self.dat_rowtype.Commit()
        self.par_rowtype = MPI.DOUBLE.Create_contiguous(self.npar)
        self.par_rowtype.Commit()

        # Scatter the data
        self.sendcnts = self.comm.gather(self.procN, root=0)
        self.comm.Scatterv([self.datArr0, (self.sendcnts, None), self.dat_rowtype],
                            self.datArr, root=0)
        self.comm.Scatterv([self.wgtArr0, (self.sendcnts, None), self.dat_rowtype],
                            self.wgtArr, root=0)

        # Repeat process for any seasonal data
        if self.data.have_seasonal:
            self.m_seas = np.empty((self.procN, self.data.npar_seasonal))
            seas_rowtype = MPI.DOUBLE.Create_contiguous(self.data.npar_seasonal)
            seas_rowtype.Commit()
            self.comm.Scatterv([self.m_seas0, (self.sendcnts, None), seas_rowtype],
                                self.m_seas, root=0)
            seas_rowtype.Free()

        # Scatter the scalar penalties
        penalties = np.empty((self.procN,))
        self.comm.Scatterv([self.penalties0, (self.sendcnts, None)], 
                            penalties, root=0)

        # Allocate arrays for model coefficients and penalty weights
        mShape = (self.procN, self.npar)
        self.m = np.zeros(mShape)
        self.λ = np.ones(mShape)
        for i in range(self.procN):
            self.λ[i,:] *= penalties[i]

        self.dat_rowtype.Free()

        return
        

    def invert(self, maxiter=1, weightingMethod='log', wtype='median', norm_tol=1.0e-4,
        nsplmod=8):

        # Choose my estimator for the weights
        if 'mean' == wtype:
            estimator = MPWeights.computeMean
        elif 'median' == wtype:
            estimator = MPWeights.computeMedian
        else:
            assert False, 'unsupported weight type. must be mean or median'

        # Instantiate a solver
        objSolver = self.solverClass(cutoff=self.cutoff, maxiter=1, eps=1.0e-2,
                                     weightingMethod=weightingMethod)

        # Make a seasonal amplitude modulator if necessary
        if self.data.have_seasonal:
            rep = [['ISPLINE',[3],[nsplmod]]]
            G_mod_ref = ts.Timefn(rep, self.data.trel)[0]
            self.nsplmod = nsplmod
        else:
            self.nsplmod = 0
        
        # Iterate
        nobs = self.nstat * self.ncomp
        for ii in range(maxiter):
            if self.rank == 0: print('At iteration', ii)
            # Loop over my portion of the data
            for jj in range(self.procN):

                # Check if a seasonal representation has been specified
                if self.data.have_seasonal:
                    G_seas = ts.Timefn(self.timeRep.rep, self.data.trel)[0]
                    template = np.dot(G_seas, self.m_seas[jj,:])
                    # Normalize the template
                    mean_spline = np.mean(template)
                    fit_norm = template - mean_spline
                    template = fit_norm / (0.5*(fit_norm.max() - fit_norm.min()))
                    G_mod = G_mod_ref.copy()
                    for nn in range(nsplmod):
                        G_mod[:,nn] *= template
                    G = np.column_stack((G_mod, self.G))
                else:
                    G = self.G

                # Get boolean indices of valid observations and weights
                dat = self.datArr[jj,:].copy()
                wgt = self.wgtArr[jj,:].copy()
                ind = np.isfinite(dat) * np.isfinite(wgt)
                
                # Subset the data and do the inversion
                Gsub, dat, wgt = G[ind,:], dat[ind], wgt[ind]
                # Get the penalties
                λ = self.λ[jj,self.cutoff:]
                # Perform estimation and store weights
                self.m[jj,:], q = objSolver.invert(dmultl(wgt,Gsub), wgt*dat, λ)
                self.λ[jj,:] = q

            # Wait for all workers to finish
            self.comm.Barrier()

            # Master gathers the results for this iteration
            self.comm.Gatherv(self.m, [self.m0, (self.sendcnts, None), self.par_rowtype], root=0)
            self.comm.Gatherv(self.λ, [self.λ0, (self.sendcnts, None), self.par_rowtype], root=0)

            # Master computes the new weights
            if self.rank == 0:
                print(' - updating weights...')
                exitFlag = self._updateWeights(estimator, norm_tol)
            else:
                exitFlag = None

            # Scatter the updated penalties
            self.comm.Scatterv([self.λ0, (self.sendcnts, None), self.par_rowtype],
                               self.λ, root=0)
            # Broadcast the exit flag
            exitFlag = self.comm.bcast(exitFlag, root=0)
            if exitFlag:
                break

        self.par_rowtype.Free()
        return


    def _updateWeights(self, estimator, norm_tol):
        """
        Update regularization penalties using distance between stations. Also check
        the exit condition for iterations.
        """

        # Temporary
        λ_new = np.zeros_like(self.λ0)
    
        # Loop over components
        for kk in range(self.ncomp):
            # Get current penalty arrays and scalar penalties for this component
            λ_comp = self.λ0[kk*self.nstat:kk*self.nstat+self.nstat,:]
            pen_comp = self.penalties0[kk*self.nstat:kk*self.nstat+self.nstat]
            # Loop over stations
            for jj in range(self.nstat):
                # Distance weights
                weights = np.tile(np.expand_dims(self.dist_weight[jj,:], axis=1), (1,self.npar))
                λ_new[kk*self.nstat+jj,:] = estimator(weights, λ_comp, pen_comp[jj])

        # Check the exit condition
        λ_diff = λ_new[:,self.cutoff:] - self.λ0[:,self.cutoff:]
        normDiff = np.linalg.norm(λ_diff, ord='fro')
        print(' - normDiff =', normDiff)
        if normDiff < norm_tol:
            exitFlag = True
        else:
            exitFlag = False

        # Copy from temporaries
        self.λ0[:,:] = λ_new

        return exitFlag


    def distributem(self, reconstruct=True, transient=False, nsplmod=8):
        """
        Distribute the solution coefficients into the data object station dictionary.
        """
        if transient:
            cutoff = self.cutoff
        else:
            cutoff = 0

        # Make a seasonal amplitude modulator if necessary
        if self.data.have_seasonal:
            rep = [['ISPLINE',[3],[nsplmod]]]
            G_mod_ref = ts.Timefn(rep, self.data.trel)[0]

        if self.rank == 0:
            ind = 0
            for component in self.components:
                for statname, stat in self.data.statGen:
                    m = self.m0[ind,:]

                    # Check if a seasonal representation has been specified
                    if self.data.have_seasonal:
                        G_seas = ts.Timefn(self.timeRep.rep, self.data.trel)[0]
                        template = np.dot(G_seas, self.m_seas0[ind,:])
                        # Normalize the template
                        mean_spline = np.mean(template)
                        fit_norm = template - mean_spline
                        template = fit_norm / (0.5*(fit_norm.max() - fit_norm.min())) 
                        G_mod = G_mod_ref.copy()
                        for nn in range(nsplmod):
                            G_mod[:,nn] *= template
                        G = np.column_stack((G_mod, self.G))
                    else:
                        G = self.G

                    recon = np.dot(G[:,cutoff:], m[cutoff:])
                    signal_steady = np.dot(G[:,nsplmod:cutoff], m[nsplmod:cutoff])
                    signal_seasonal = np.dot(G[:,:nsplmod], m[:nsplmod])
                    signal_remove = signal_steady + signal_seasonal

                    #recon = np.dot(G[:,:nsplmod], m[:nsplmod])
                    #signal_remove = np.dot(G[:,nsplmod:], m[nsplmod:])

                    try:
                        setattr(stat, 'm_' + component, self.m0[ind,:])
                        if reconstruct:
                            dat = getattr(stat, component)
                            setattr(stat, 'filt_' + component, recon)
                            setattr(stat, component, dat - signal_remove)
                            setattr(stat, 'steady_' + component, signal_remove)
                    except AttributeError:
                        stat['m_' + component] = self.m0[ind,:]
                        if reconstruct:
                            dat = np.array(stat[component])
                            stat['filt_' + component] = recon
                            stat[component] = dat - signal_remove
                            stat['steady_' + component] = signal_steady
                            stat['seasonal_' + component] = signal_seasonal
                    ind += 1
        return

# end of file