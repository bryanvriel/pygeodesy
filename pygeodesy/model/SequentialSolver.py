#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import tsinsar as ts
from .Solver import Solver
from .MPClasses import MPWeights

dmultl = ts.dmultl

class SequentialSolver(Solver):
    """
    Perform weighted inversions sequentially.
    """

    def __init__(self, data, timeRep, solver, penaltyDict, smooth=1.0, L0=None,
                 spatial_weighting=False):
        """
        Initialize the parent Solver class and get data arrays.
        """
        super().__init__(data, timeRep, solver)

        # Get data arrays
        self.datArr, self.wgtArr = self.data.getDataArrays(order='rows')
        assert self.ndat == self.datArr.shape[1]

        # Compute distance weights
        self.spatialFlag = spatial_weighting
        if self.spatialFlag:
            self.dist_weight = self.data.computeNetworkWeighting(smooth=smooth, L0=L0)

        # Make 1-d array of penalties
        self.penalties = np.ones((self.nstat*self.ncomp,))
        ind_beg = 0
        for component in data.components:
            ind_end = ind_beg + self.nstat
            self.penalties[ind_beg:ind_end] *= penaltyDict[component]
            ind_beg += self.nstat

        # Allocate arrays for model coefficients and penalty weights
        N = self.nstat * self.ncomp
        self.m = np.zeros((N, self.npar))
        self.λ = np.ones((N, self.npar))
        for i in range(N):
            self.λ[i,:] *= self.penalties[i]
       
        return

    def invert(self, maxiter=1, weightingMethod='log', wtype='median', norm_tol=1.0e-4):

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

        # Iterate
        N = self.nstat * self.ncomp
        exitFlag = False
        for ii in range(maxiter):

            # Loop over the data
            for jj in range(N):
                # Get boolean indices of valid observations and weights
                dat = self.datArr[jj,:].copy()
                wgt = self.wgtArr[jj,:].copy()
                ind = np.isfinite(dat) * np.isfinite(wgt)
                Gsub, dat, wgt = self.G[ind,:], dat[ind], wgt[ind]
                # Get the penalties
                λ = self.λ[jj,self.cutoff:]
                # Perform estimation and store weights
                self.m[jj,:], q = objSolver.invert(dmultl(wgt,Gsub), wgt*dat, λ)
                self.λ[jj,:] = q

            # Update weights
            if self.spatialFlag:
                exitFlag = self._updateWeights(estimator, norm_tol)
            if exitFlag: 
                break

        return

    
    def _updateWeights(self, estimator, norm_tol):
        """
        Update regularization penalties using distance between stations. Also check
        the exit condition for iterations.
        """

        # Temporary
        λ_new = np.zeros_like(self.λ)

        # Loop over all stations and components
        N = self.nstat * self.ncomp
        for jj in range(N):
            # Distance weights
            weights = np.tile(np.expand_dims(self.dist_weight[jj,:], axis=1), (1,self.npar))
            # Compute distance-weighted penalty
            λ_new[jj,:] = estimator(weights, self.λ, self.penalties[jj])

        # Check the exit condition
        λ_diff = λ_new - self.λ
        normDiff = np.linalg.norm(λ_diff, ord='fro')
        print(' - normDiff =', normDiff)
        if normDiff < norm_tol:
            exitFlag = True
        else:
            exitFlag = False

        # Copy from temporaries
        self.λ[:,:] = λ_new

        return exitFlag


    def distributem(self):
        """
        Distribute the solution coefficients into the data object station dictionary.
        """
        ind = 0
        for component in self.components:
            for statname in self.data.name:
                stat = self.data.statDict[statname]
                setattr(stat, 'm_' + component, self.m[ind,:])
                ind += 1
        return

# end of file
