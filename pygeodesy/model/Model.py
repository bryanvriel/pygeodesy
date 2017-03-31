#-*- coding: utf-8 -*-

import numpy as np
from datetime import datetime
from giant.utilities import timefn
import sys


class Model:
    """
    Class for handling time series predictions.
    """

    def __init__(self, t, collection=None, t0=None, tf=None, rank=0):
        """
        Initialize the Model class with a TimeRepresentation object.

        Parameters
        ----------
        t: array type
            Array of observation times as datetime objects
        collection: {giant.utilities.timefn.TimefnCollection, None}, optional
            GIAnT TimefnCollection instance. If None, constructs a polynomial collection.
        t0: datetime or None
            Starting date for estimating parameters.
        tf: datetime or None
            Ending date for estimating parameters.
        rank: int, optional
            MPI rank. Default: 1.
        """

        if isinstance(collection, timefn.TimefnCollection):
            self.collection = collection
        else:
            self.collection = timefn.TimefnCollection()
            self.collection.append(timefn.fnmap['poly'](t[0], order=1, units='years'))

        # Evaluate the collection
        self.G = self.collection(t)

        # Get indices for the functional partitions
        fnParts = timefn.getFunctionTypes(self.collection)
        for key in ('secular', 'seasonal', 'transient', 'step'):
            setattr(self, 'i%s' % key, fnParts[key])
        self.npar = self.G.shape[1]
        self.ifull = np.arange(self.npar, dtype=int)
        self._updatePartitionSizes()

        # Make mask for time window for estimating parameters
        t0 = datetime.strptime(t0, '%Y-%m-%d') if t0 is not None else t[0]
        tf = datetime.strptime(tf, '%Y-%m-%d') if tf is not None else t[-1]
        self.time_mask = (t >= t0) * (t <= tf)

        # Save the regularization indices
        self.reg_indices = fnParts['reg']
    
        # Save MPI rank
        self.rank = rank

        return


    def _updatePartitionSizes(self):
        """
        Update the sizes of the list of indices for the functional partitions.
        """
        for attr in ('secular', 'seasonal', 'transient', 'step', 'full'):
            ind_list = getattr(self, 'i%s' % attr)
            setattr(self, 'n%s' % attr, len(ind_list))
        return


    def invert(self, solver, d, wgt=None):
        """
        Perform least squares inversion using a solver.
        """

        # Do the inversion
        ind = np.isfinite(d) * self.time_mask
        if wgt is None:
            m, Cm = solver.invert(self.G[ind,:], d[ind])
        else:
            m, Cm = solver.invert(self.G[ind,:], d[ind], wgt=wgt[ind])

        # Save the partitions
        self.Cm = Cm
        self.coeff = {'secular': m[self.isecular], 'seasonal': m[self.iseasonal],
            'transient': m[self.itransient], 'step': m[self.istep]}
        return m


    def initializeOutput(self):
        """
        Makes an empty dictionary for holding partitional results.
        """
        N = self.G.shape[0]
        zero = np.zeros((N,))
        self.fit_dict = {}
        for attr in ('step', 'seasonal', 'secular', 'transient', 'full', 'sigma'):
            self.fit_dict[attr] = zero.copy()
        return


    def updateOutput(self, new_dict):
        """
        Update functional fits using a given dictionary.
        """
        for key,arr in new_dict.items():
            self.fit_dict[key] += arr
        return


    def computeSeasonalAmpPhase(self):
        """
        Try to compute the seasonal amplitude and phase.
        """
        try:
            m1, m2 = self.coeff['seasonal'][-2:]
            phs = np.arctan2(m2, m1) * 182.5/np.pi
            amp = np.sqrt(m1**2 + m2**2)
            if phs < 0.0:
                phs += 365.0
        except ValueError:
            phs, amp = None, None
        return amp, phs

    
    def getSecular(self, mvec):
        """
        Return the polynomial component with the highest power.
        """
        msec = mvec[self.isecular]
        variance_secular = np.diag(self.Cm)[self.isecular]
        if len(msec) != 2:
            return 0.0, 0.0
        else:
            return msec[-1], np.sqrt(variance_secular[-1])


    def getStep(self, mvec):
        """
        Return step coefficients.
        """
        mstep = mvec[self.istep]
        variance_step = np.diag(self.Cm)[self.isecular]
        if len(msec) < 1:
            return 0.0, 0.0
        else:
            return mstep[-1], variance_step[-1]


    def predict(self, mvec, out=None, sigma=True):
        """
        Predict time series with a functional decomposition specified by data.

        Parameters
        ----------
        mvec: np.ndarray
            Array of parameters.
        """

        # Compute different components
        secular = np.dot(self.G[:,self.isecular], mvec[self.isecular])
        seasonal = np.dot(self.G[:,self.iseasonal], mvec[self.iseasonal])
        transient = np.dot(self.G[:,self.itransient], mvec[self.itransient])
        step = np.dot(self.G[:,self.istep], mvec[self.istep])

        # Compute the functional partitions
        results = {'secular': secular, 'seasonal': seasonal, 'transient': transient,
            'step': step, 'full': secular + seasonal + transient + step}

        # Add uncertainty if applicable
        if hasattr(self, 'Cm') and sigma:
            sigma = np.sqrt(np.diag(np.dot(self.G, np.dot(self.Cm, self.G.T))))
            results['sigma'] = sigma

        return results


    def detrend(self, d, recon, parts_to_remove):
        """
        Detrend the data and update model fit.
        """
        # Determine which functional partition to keep
        parts_to_keep = np.setdiff1d(['seasonal', 'secular', 'transient', 'step'], 
            parts_to_remove)

        # Compute signal to remove
        signal_remove = np.zeros_like(d)
        for key in parts_to_remove:
            signal_remove += recon[key]

        # And signal to keep
        signal_keep = np.zeros_like(d)
        for key in parts_to_keep:
            signal_keep += recon[key]
        
        # Detrend
        d -= signal_remove
        return signal_keep


# end of file
