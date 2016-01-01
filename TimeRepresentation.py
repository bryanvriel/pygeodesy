#-*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import tsinsar as ts


class TimeRepresentation:
    """
    Class for parameterizing temporal evolution functions.
    """

    def __init__(self, t, rep=None, cutoff=None, G=None):

        self.t = t
        self.rep = rep
        self.cutoff = cutoff

        self._matrix = self._seas_matrix = None
        self.npar = None

        if self.rep is not None:
            self._rep2matrix()

        # If a matrix is provided, go ahead and store to _matrix
        if G is not None:
            self._matrix = G
            self.npar = G.shape[1]

        self.repDict = None
        self.repKeys = []

        return


    def addEntry(self, *args):
        """
        Add a time rep entry to dictionary.
        """

        nargs = len(args)
        assert nargs > 1, 'Not enough arguments supplied to addEntry'
        key = args[0]

        if self.repDict is None:
            self.repDict = {
        }
        if nargs > 2:
            values = args[1:]
        else:
            values = args[1]
        if 'ispline' in key:
            tk = np.linspace(self.t[0], self.t[-1], values[1][0])
            dtk = 2.0 * (tk[1] - tk[0])
            print('Spline time:', dtk)
        self.repDict[key.upper()] = values
        self.repKeys.append(key.upper())
        return


    def deleteEntry(self, key):
        """
        Delete a time rep entry from dictionary.
        """
        if key in self.repKeys:
            del self.repDict[key]
            self.repKeys.remove(key)
        return


    def tolist(self):
        """
        Construct GIAnT-compatible list of time series representation.
        """
        assert self.repDict is not None, 'Must construct time dictionary first'

        rep = []
        for key in self.repKeys:
            values = self.repDict[key]
            repType = key.upper()
            if 'STEP' in repType:
                repType = 'STEP'
            elif 'EXP' in repType:
                repType = 'EXP'
            elif 'LOG' in repType:
                repType = 'LOG'
            elif 'ISPLINE' in repType:
                repType = 'ISPLINE'

            if type(values) is list:
                entry = [repType, values]
            else:
                entry = [repType]
                for value in values:
                    entry.append(value)
            rep.append(entry)
        return rep


    def getFunctionalPartitions(self, returnstep=False):
        """
        Return the indices of the partitions.
        """

        steady = []; seasonal = []; post = []; step = []
        current = 0
        for cnt, inkey in enumerate(self.repKeys):

            key = inkey.lower()
            if 'poly' in key:
                steady.extend([current, current+1])
                current += 2
            elif 'step' in key:
                steady.extend([current])
                step.extend([current])
                current += 1
            elif 'exp' in key:
                post.extend([current])
                current += 1
            elif 'log' in key:
                post.extend([current])
                current += 1
            elif 'seasonal' in key:
                seasonal.extend([current, current+1, current+2, current+3])
                current += 4
            elif 'ispline' in key:
                nspl = self.repDict[key.upper()][1][0]
                post.extend((current + np.arange(nspl, dtype=int)).tolist())
                current += nspl

        if returnstep:
            return steady, seasonal, post, step
        else:
            return steady, seasonal, post


    def maskRepresentation(self, mask):
        """
        Given a mask with the same size as self.t, we remove any entries from the
        dictionary that are not constrained by the data.
        """

        assert len(mask) == len(self.t), 'incompatible mask size'

        # Get time of first valid observation
        valid_times = self.t[mask]
        t0, tf = valid_times[0], valid_times[-1]

        # Loop over keys
        keepKeys = []
        for key in copy.deepcopy(self.repKeys):

            keep = True
            values = self.repDict[key]
            repType = key.upper()
            if 'STEP' in repType:
                repType = 'STEP'
            elif 'EXP' in repType:
                repType = 'EXP'
            elif 'LOG' in repType:
                repType = 'LOG'
            elif 'ISPLINE' in repType:
                repType = 'ISPLINE'

            if repType in ['STEP', 'EXP', 'LOG']:

                # Extract t0
                if repType == 'STEP':
                    toff = values[0]
                else:
                    toff = values[0][0]

                # Test it
                if toff < t0:
                    keep = False
                if toff > tf:
                    keep = False

            if not keep:
                self.deleteEntry(key)
            else:
                keepKeys.append(key)

        self.repKeys = keepKeys
        return


    def setRepresentation(self, rep):
        """
        Store the GIAnT-style representation.
        """
        assert type(rep) is list, 'input representation must be a list'
        self.rep = rep
        self._rep2matrix()
        return


    def setCutoff(self, cutoff):
        """
        Set the cutoff for partitioning between non-regularized and regularized coefficients.
        """
        self.cutoff = cutoff
        return


    def setRegIndices(self, regF):
        """
        Use regularization values to set which indices will be regulated.
        """
        self.noreg_ind = (regF < 0.9).nonzero()[0]
        self.reg_ind = (regF > 0.1).nonzero()[0]
        self.cutoff = len(noreg_ind)
        return


    def partitionMatrix(self, cutoff=None):
        """
        Partition the time representation matrix into non-reg and reg coefficients.
        """
        # Determine or unpack regularization indices
        if cutoff is not None and self.cutoff is None:
            noreg_ind = range(cutoff)
            reg_ind = np.setxor1d(range(self.npar), noreg_ind)
        else:
            noreg_ind, reg_ind = self.noreg_ind, self.reg_ind
        # Slice the matrix
        self._matrixNoReg = self._matrix[:,noreg_ind]
        self._matrixReg = self._matrix[:,reg_ind]
        return


    def getObs(self, index, statInd=0):
        """
        Get a slice of the time representation matrix at a given observation index.
        """
        # Make matrix if needed
        if self._matrix is None and self.rep is not None:
            self._rep2matrix()
        # Get matrix slice
        kmat = self._getMatrixObs(index)
        # Get seasonal matrix slice
        kseas = self._getSeasMatrixObs(index, statInd)
        # Combine the two
        return np.hstack((kseas, kmat))


    def _getMatrixObs(self, index):
        """
        A hidden helper function to slice the representation matrix for a given
        observation index. If no matrix exists, just returns an empty list.
        """
        if self._matrix is None and self.rep is None:
            return []
        else:
            return self._matrix[index,:]

    
    def _getSeasMatrixObs(self, index, statInd):
        """
        A hidden helper function to slice the seasonal matrix for a given
        observation index. If no matrix exists, just returns an empty list.
        """
        if self._seas_matrix is None:
            return []
        else:
            return self._seas_matrix[index,:,statInd]


    def _rep2matrix(self):
        """
        Convert the string representation to a numpy array
        """
        # Use Timefn to get matrix
        self._matrix, mName, regF = ts.Timefn(self.rep, self.t-self.t[0])
        # Determine indices for non-regularized variables
        self.noreg_ind = (regF < 0.9).nonzero()[0]
        # And indices for regularized variables
        self.reg_ind = (regF > 0.1).nonzero()[0]
        # Store number of parameters
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
        print('Warning: setting TimeRepresentation.matrix explicitly')
        self._matrix = mat
        return


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


    @property
    def trel(self):
        return self.t - self.t[0]
    @trel.setter
    def trel(self, val):
        raise NotImplementedError('Cannot set trel explicitly')


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
