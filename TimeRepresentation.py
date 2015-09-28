#-*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import tsinsar as ts


class TimeRepresentation:
    """
    Class for parameterizing temporal evolution functions.
    """

    def __init__(self, t, rep=None, cutoff=0, G=None):

        self.t = t
        self.rep = rep
        self.cutoff = cutoff

        self._matrix = None
        self.npar = None

        if self.rep is not None:
            self._rep2matrix()

        # If a matrix is provided, go ahead and store to _matrix
        if G is not None:
            self._matrix = G
            self.npar = G.shape[1]

        return


    def setRepresentation(self, rep):
        """
        Store the GIAnT-style representation.
        """
        assert  type(rep) is list, 'input representation must be a list'
        self.rep = rep
        self._rep2matrix()
        return


    def setCutoff(self, cutoff):
        """
        Set the cutoff for partitioning between non-regularized and regularized coefficients.
        """
        self.cutoff = cutoff
        return


    def partitionMatrix(self, cutoff=None):
        """
        Partition the time representation matrix into non-reg and reg coefficients.
        """
        if cutoff is not None and self.cutoff is None:
            self.cutoff = cutoff
        if self.cutoff > 0:
            self._matrixNoReg = self._matrix[:,:self.cutoff]
        else:
            self._matrixNoReg = None
        self._matrixReg = self._matrix[:,self.cutoff:]
        return


    def _rep2matrix(self):
        """
        Convert the string representation to a numpy array
        """
        self._matrix = np.asarray(ts.Timefn(self.rep, self.t-self.t[0])[0], order='C')
        self.npar = self._matrix.shape[1]
        return 

    # --------------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------------

    @property
    def matrix(self):
        if self._matrix is None:
            assert self.rep is not None
            self._rep2matrix()
        return self._matrix
    @matrix.setter
    def matrix(self, mat):
        raise NotImplementedError('Cannot set matrix explicitly')


    @property
    def matrixNoReg(self):
        return self._matrixNoReg
    @matrixNoReg.setter
    def matrixNoReg(self, mat):
        raise NotImplementedError('Cannot set matrix explicitly')


    @property
    def matrixReg(self):
        return self._matrixReg
    @matrixReg.setter
    def matrixReg(self, mat):
        raise NotImplementedError('Cannot set matrix explicitly')


def makeRepresentation(tdec, rank, splineOrder=3, secular=False, maxspl=2048):
    """
    Constructs a highly overcomplete dictionary for approximately shift-invariant sparsity.
    """

    # Check that the size of tdec is a power of two
    N = tdec.size
    assert np.log2(N) % int(np.log2(N)) < 1.0e-8, \
        'Size of input data is not a power of two'

    # Determine timescales to fill in G
    N_levels = int(np.log2(N))
    levels = 0
    istart = 2
    for j in range(N_levels-2, -1, -1):
        nspl = 2**(j + 2)
        if nspl > maxspl:
            istart += 1
            continue
        levels += 1

    # Now construct G
    G = np.zeros((N, N*levels))
    cnt = 0
    for j in range(N_levels-istart, -1, -1):

        # Determine timescale of I-spline for current dyadic level
        nspl = 2**(j + 2)
        if nspl > maxspl:
            cnt += 1
            continue
        tk = np.linspace(tdec.min(), tdec.max(), nspl)
        dtk = abs(tk[1] - tk[0])
        if rank == 0:
            print('For', nspl, 'splines, tau is', 2*dtk, 'years or', 2*dtk*365.0, 'days')

        # Loop over time shifts corresponding to every observation epoch
        for p in range(N):
            hfn = ispline(splineOrder, dtk, tdec - tdec[p])
            G[:,N*cnt+p] = hfn

        cnt += 1

    # Add secular components if requested
    if secular:
        rep = [['OFFSET',[0.0]], ['LINEAR',[0.0]]]
        Gsec = np.asarray(ts.Timefn(rep, tdec-tdec[0])[0], order='C')
        G = np.column_stack((Gsec, G))

    return G


# end of file    
