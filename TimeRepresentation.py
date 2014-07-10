#-*- coding: utf-8 -*-

import numpy as np
import tsinsar as ts


class TimeRepresentation:
    """
    Class for parameterizing temporal evolution functions.
    """

    def __init__(self, t, rep=None, cutoff=0):

        self.t = t
        self.rep = rep
        self.cutoff = cutoff

        self._matrix = None
        self.npar = None

        if self.rep is not None:
            self._rep2matrix()

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



# end of file    
