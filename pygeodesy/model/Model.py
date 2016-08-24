#-*- coding: utf-8 -*-

import numpy as np
from giant.utilities import timefn
import sys


class Model:
    """
    Class for handling time series predictions.
    """

    def __init__(self, t, collection=None, rank=0):
        """
        Initialize the Model class with a TimeRepresentation object.

        Parameters
        ----------
        t: array type
            Array of observation times as datetime objects
        collection: {giant.utilities.timefn.TimefnCollection, None}, optional
            GIAnT TimefnCollection instance. If None, constructs a polynomial collection.
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
        ind = np.isfinite(d)
        if wgt is None:
            m, Cm = solver.invert(self.G[ind,:], d[ind])
        else:
            m, Cm = solver.invert(self.G[ind,:], d[ind], wgt=wgt[ind])

        # Save the partitions
        self.Cm = Cm
        self.coeff = {'secular': m[self.isecular], 'seasonal': m[self.iseasonal],
            'transient': m[self.itransient]}
        return m


    def initializeOutput(self):
        """
        Makes an empty dictionary for holding partitional results.
        """
        N = self.G.shape[0]
        zero = np.zeros((N,))
        self.fit_dict = {}
        for attr in ('seasonal', 'secular', 'transient', 'full', 'sigma'):
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


    def predict(self, mvec, out=None):
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

        # Compute the functional partitions
        results = {'secular': secular, 'seasonal': seasonal, 'transient': transient,
                'full': secular + seasonal + transient}

        # Add uncertainty if applicable
        if hasattr(self, 'Cm'):
            sigma = np.sqrt(np.diag(np.dot(self.G, np.dot(self.Cm, self.G.T))))
            results['sigma'] = sigma

        return results


    def detrend(self, d, recon, parts_to_remove):
        """
        Detrend the data and update model fit.
        """
        # Determine which functional partition to keep
        parts_to_keep = np.setdiff1d(['seasonal', 'secular', 'transient'], parts_to_remove)

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
