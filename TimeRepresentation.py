#-*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import tsinsar as ts
import copy


class TimeRepresentation:
    """
    Class for parameterizing temporal evolution functions.
    """

    def __init__(self, t, rep=None, cutoff=None, G=None, Jmat=None, rank=0):

        self.t = t
        self.rep = rep
        self.cutoff = cutoff
        self.rank = rank

        self._matrix = self._seas_matrix = None
        self.nobs = len(t)
        self.nseas_par = 0
        self.nstat = 1

        if self.rep is not None:
            self._rep2matrix()

        # If a matrix is provided, go ahead and store to _matrix
        if G is not None:
            self._matrix = G

        self.Jmat = Jmat
        #if Jmat is not None:
        #    self.nobs = Jmat.shape[0]

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
            self.repDict = {}
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


    def fromlist(self, replist):
        """
        Transfer time rep info from GIAnT-compatible list to self.
        """
        self.repKeys = []
        self.repDict = {}
        modifier = 0
        for entry in replist:
            key = entry[0]
            # Make sure key is unique
            if key in self.repKeys:
                key = '%s_%03d' % (key, modifier)
                modifier += 1
            # Add entry
            self.repDict[key] = entry[1:]
            self.repKeys.append(key)
        self.rep = replist
        self._rep2matrix()
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
            elif 'PBSPLINE' in repType:
                repType = 'PBSPLINE'

            if type(values) is list:
                entry = [repType, values]
            else:
                entry = [repType]
                for value in values:
                    entry.append(value)
            rep.append(entry)
        self.rep = rep
        self._rep2matrix()
        return rep


    def seasonalModulation(self, seasonalfile):
        """
        Open an Insar stack containing seasonal signals for every station/pixel. 
        This will be used to create modulated seasonal matrix.
        """
        # First build the modulation design matrix
        for key in self.repKeys:
            values = self.repDict[key]
            repType = key.upper()
            if 'modulated' in key:
                rep = [['ISPLINE', values]]
                seas_matrix = ts.Timefn(rep, self.t)[0]
                break

        # Load an Insar stack for the seasonal data
        comm = comm or MPI.COMM_WORLD
        data = Insar(name='seasonal', comm=comm)

        # Now form 3D modulated matrix



    def getFunctionalPartitions(self, returnstep=False):
        """
        Return the indices of the partitions.
        """

        secular = []; seasonal = []; transient = []; step = []
        current = 0
        for cnt, inkey in enumerate(self.repKeys):

            key = inkey.lower()
            if 'poly' in key:
                npoly = self.repDict[key.upper()][0][0] + 1
                secular.extend((current + np.arange(npoly, dtype=int)).tolist())
                current += npoly
            elif 'step' in key:
                secular.extend([current])
                step.extend([current])
                current += 1
            elif 'exp' in key:
                transient.extend([current])
                current += 1
            elif 'log' in key:
                transient.extend([current])
                current += 1
            elif 'seasonal' in key:
                seasonal.extend([current, current+1, current+2, current+3])
                current += 4
            elif 'pbspline' in key:
                nspl = self.repDict[key.upper()][1][0]
                seasonal.extend((current + np.arange(nspl, dtype=int)).tolist())
                current += nspl
            elif 'modulated' in key:
                nspl = self.repDict[key.upper()][1][0]
                seasonal.extend((current + np.arange(nspl, dtype=int)).tolist())
                current += nspl
            elif 'ispline' in key:
                nspl = self.repDict[key.upper()][1][0]
                transient.extend((current + np.arange(nspl, dtype=int)).tolist())
                current += nspl

        # Save the size of the arrays
        self.nsecular = len(secular)
        self.nseasonal = len(seasonal)
        self.ntransient = len(transient)
        self.nstep = len(step)
        self.nfull = self.npar

        if returnstep:
            return secular, seasonal, transient, step
        else:
            return secular, seasonal, transient


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
        # Use regF values to determine if parameter is regularized
        noreg_ind = (regF < 0.9).nonzero()[0]
        reg_ind = (regF > 0.1).nonzero()[0]
        # Modify regularization indices for any seasonal components
        self.noreg_ind = noreg_ind[self.nseas_par:] - self.nseas_par
        self.reg_ind = reg_ind - self.nseas_par
        # Save "cutoff" value
        self.cutoff = len(self.noreg_ind)
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
        if len(kseas) > 0:
            # Combine the two
            return np.hstack((kseas, kmat))
        else:
            return kmat


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

    
    def predict(self, x, ptype='all', in_nstat=None):
        """
        Makes predictions given a vector of coefficients. 'ptype' specifies the
        predictions to return:
            'all': secular, seasonal, transient
            'seasonal': seasonal only
            'secular': secular only
            'transient': transient only

        Right now, this is meant to only work with coefficients generated by
        ADMMSolver, where problem is split among features. Therefore, the length
        of the prediction arrays is Ntime x Nstat.
        """ 
        # Initialize prediction arrays
        nstat = in_nstat or self.nstat
        seasonal = np.zeros((self.nobs*nstat,))
        secular = np.zeros((self.nobs*nstat,))
        transient = np.zeros((self.nobs*nstat,))

        # Retrieve regularization indices
        noreg_ind, reg_ind = self.noreg_ind, self.reg_ind

        # Make predictions on a station-by-station basis
        for k in range(nstat):
            # Retrieve coefficient vector corresponding to current station
            xi_full = x[k::nstat]
            # Take out seasonal coefficients separately
            xi_seas = xi_full[:self.nseas_par]
            xi = xi_full[self.nseas_par:]
            # Predictions using standard design matrix
            if self._matrix is not None:
                if len(noreg_ind) > 0:
                    secular[k::nstat] = np.dot(self._matrix[:,noreg_ind], xi[noreg_ind])
                if len(reg_ind) > 0:
                    transient[k::nstat] = np.dot(self._matrix[:,reg_ind], xi[reg_ind])
            # Predictions using modulated seasonal design matrix
            if self._seas_matrix is not None:
                # Retrieve seasonal matrix for current station
                seas_mat = self._seas_matrix[:,:,k]
                seasonal[k::nstat] = np.dot(seas_mat, xi_seas)

        # Make a temporary output dictionary
        outdict = {'secular': secular, 'seasonal': seasonal, 'transient': transient,
            'total': secular + seasonal + transient}

        # Return specified keys
        if ptype == 'all':
            return outdict
        else:
            try:
                output = {ptype: outdict[ptype]}
            except KeyError:
                print('Unsupported prediction type. Returning total.')
                output = {'total': outdict['total']}
            return output
            

    def _rep2matrix(self):
        """
        Convert the string representation to a numpy array
        """
        # Use Timefn to get matrix
        self._matrix, mName, regF = ts.Timefn(self.rep, self.t-self.t[0])
        self.regF = regF
        # Modulate by a connectivity matrix (for InSAR)
        if self.Jmat is not None:
            self._matrix = np.dot(self.Jmat, self._matrix)
        # Determine indices for non-regularized variables
        self.noreg_ind = (regF < 0.9).nonzero()[0]
        # And indices for regularized variables
        self.reg_ind = (regF > 0.1).nonzero()[0]

        return 

    # --------------------------------------------------------------------------------
    # Properties
    # --------------------------------------------------------------------------------

    @property
    def npar(self):
        n = 0
        if self._matrix is not None:
            n += self._matrix.shape[1]
        if self._seas_matrix is not None:
            n += self._seas_matrix.shape[1]
        return n
    @npar.setter
    def npar(self, value):
        raise NotImplementedError('Cannot set npar explicitly')

    # Generic temporal design matrix
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


    # Seasonal modulation design matrix
    @property
    def seasonal_matrix(self):
        return self._seas_matrix
    @seasonal_matrix.setter
    def seasonal_matrix(self, mat):
        self._seas_matrix = mat
        self.nseas_par, self.nstat = mat.shape[1:]
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
