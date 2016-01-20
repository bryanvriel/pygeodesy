#-*- coding: utf-8 -*-

import numpy as np

class Model:
    """
    Class for handling time series predictions.
    """

    def __init__(self, rep, rank=0, Jmat=None):
        """
        Initialize the Model class with a TimeRepresentation object.

        Parameters
        ----------
        rep: TimeRepresentation
            TimeRepresentation object.
        rank: int, optional
            MPI rank. Default: 1.
        Jmat: {None, ndarray}, optional
            Optional connectivity matrix to pre-multiply design matrix.
        """
        # Get the design matrices
        self.rep = rep
        self.H = rep.matrix
        if Jmat is not None:
            self.G = np.dot(Jmat, self.H)
        self.npar = self.H.shape[1]

        # Get indices for the functional partitions
        self.isecular, self.iseasonal, self.itransient, self.istep = [indices
            for indices in rep.getFunctionalPartitions(returnstep=True)] 
        self.ifull = np.arange(self.npar, dtype=int)

        # Save the regularization indices
        self.reg_indices = rep.reg_ind

        # Save MPI rank
        self.rank = rank

        return


    def predict(self, mvec, data, chunk):
        """
        Predict time series with a functional decomposition specified by data.

        Parameters
        ----------
        mvec: ndarray
            Array of shape (N,Ny,Nx) representation chunk of parameters.
        data: dict
            Dictionary of {funcString: dataObj} pairs where funcString is a str
            in ['full', 'secular', 'seasonal', 'transient'] specifying which
            functional form to reconstruct, and dataObj is an Insar object to
            with an appropriate H5 file to store the reconstruction.
        chunk: list
            List of [slice_y, slice_x] representing chunk parameters.
        """
        # Only master does any work
        if self.rank == 0:

            # Consistency check
            Nt,Ny,Nx = mvec.shape
            assert Nt == self.npar, 'Inconsistent dimension for mvec and design matrix.'

            # Perform prediction component by component
            out = {
                'secular': np.einsum('ij,jmn->imn', self.H[:,self.isecular], 
                    mvec[self.isecular,:,:]),
                'seasonal': np.einsum('ij,jmn->imn', self.H[:,self.iseasonal], 
                    mvec[self.iseasonal,:,:]),
                'transient': np.einsum('ij,jmn->imn', self.H[:,self.itransient], 
                    mvec[self.itransient,:,:])
            }
            out['full'] = out['secular'] + out['seasonal'] + out['transient']

            # Loop over the function strings and data objects
            for key, dataObj in data.items():
                # Write out the prediction
                dataObj.setChunk(out[key], chunk[0], chunk[1], dtype='recon') 
                # And the parameters
                ind = getattr(self, 'i%s' % key)
                dataObj.setChunk(mvec[ind,:,:], chunk[0], chunk[1], dtype='par')

        return


# end of file
