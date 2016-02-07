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
        self.Jmat = Jmat
        self.Ntime, self.npar = self.H.shape
        
        # Initial ownership range is the full range
        self.jstart, self.jend = 0, self.npar

        # Get indices for the functional partitions
        self.isecular, self.iseasonal, self.itransient, self.istep = [indices
            for indices in rep.getFunctionalPartitions(returnstep=True)] 
        self.ifull = np.arange(self.npar, dtype=int)
        # And save the list sizes
        self._updatePartitionSizes()
        
        # Save the regularization indices
        self.reg_indices = rep.reg_ind

        # Save MPI rank
        self.rank = rank

        return


    def setOwnershipRange(self, jstart, jend):
        """
        Set the range of temporal parameters to use.

        Parameters
        ----------
        jstart: int
            Starting index.
        jend: int
            Ending index, non-inclusive.
        """
        self.jstart, self.jend = jstart, jend
        return


    def addModulatingSplines(self, nsplmod):
        """
        Adjust indices for functional partitions for additional modulating splines.
        We assume these splines are arranged first in the parameter vector.

        Parameters
        ----------
        nsplmod: int
            Number of modulating splines to add.
        """
        # Skip if nsplmod == 0
        if nsplmod == 0:
            return

        # First adjust the indices we already have
        for attr in ('isecular', 'iseasonal', 'itransient', 'istep'):
            ilist = getattr(self, attr)
            newlist = [i + nsplmod for i in ilist]
            setattr(self, attr, newlist)

        # Pre-pend new seasonal indices
        self.iseasonal = list(range(nsplmod)) + self.iseasonal
        self.npar += nsplmod
        self.ifull = np.arange(self.npar, dtype=int)

        # Update partition sizes
        self._updatePartitionSizes()

        # Pre-pend columns of zeros for design matrix
        self.H = np.column_stack((np.zeros((self.Ntime,nsplmod), 
            dtype=self.H.dtype), self.H))
        return


    def _updatePartitionSizes(self):
        """
        Update the sizes of the list of indices for the functional partitions.
        """
        for attr in ('secular', 'seasonal', 'transient', 'step', 'full'):
            ind_list = getattr(self, 'i%s' % attr)
            setattr(self, 'n%s' % attr, len(ind_list))
        return


    def predict(self, mvec, data, chunk, Gmod=None, Amod=None):
        """
        Predict time series with a functional decomposition specified by data.

        Parameters
        ----------
        mvec: np.ndarray
            Array of shape (N,Ny,Nx) representation chunk of parameters.
        data: dict
            Dictionary of {funcString: dataObj} pairs where funcString is a str
            in ['full', 'secular', 'seasonal', 'transient'] specifying which
            functional form to reconstruct, and dataObj is an Insar object to
            with an appropriate H5 file to store the reconstruction.
        chunk: list
            List of [slice_y, slice_x] representing chunk parameters.
        Gmod: np.ndarray, optional
            Modulated seasonal design matrix.
        Amod: np.ndarray, optional
            Amplitude modulation of seasonal design matrix.
        """
        # Only master does any work
        if self.rank == 0:

            # Consistency check
            Nt,Ny,Nx = mvec.shape
            assert Nt == self.npar, 'Inconsistent dimension for mvec and design matrix.'

            # Compute secular signal
            secular = np.einsum('ij,jmn->imn', self.H[:,self.isecular], 
                    mvec[self.isecular,:,:])
            
            # Compute seasonal signal
            if Gmod is None:
                seasonal = np.einsum('ij,jmn->imn', self.H[:,self.iseasonal], 
                    mvec[self.iseasonal,:,:])
            else:
                seasonal = np.einsum('ijmn,jmn->imn', Gmod, mvec[self.iseasonal,:,:])

            # Compute transient
            transient = np.einsum('ij,jmn->imn', self.H[:,self.itransient], 
                    mvec[self.itransient,:,:])

            # Save decomposition in dictionary
            out = {'secular': secular, 'seasonal': seasonal, 'transient': transient,
                'full': secular + seasonal + transient}

            # Add modulation if requested
            if Amod is not None and 'mod' in data.keys():
                out['mod'] = np.einsum('ijmn,jmn->imn', Amod, mvec[self.iseasonal,:,:])

            # Also make output for insar if Jmat is saved
            if self.Jmat is not None and 'insar' in data.keys():
                out['insar'] = np.einsum('ij,jmn->imn', self.Jmat, out['full'])

            # Loop over the function strings and data objects
            for key, dataObj in data.items():
                # Write out the prediction
                if key == 'insar':
                    dataObj.setChunk(out[key], chunk[0], chunk[1]) 
                else:
                    dataObj.setChunk(out[key], chunk[0], chunk[1], dtype='recon') 
                    # And the parameters
                    ind = getattr(self, 'i%s' % key)
                    dataObj.setChunk(mvec[ind,:,:], chunk[0], chunk[1], dtype='par')

        return


# end of file
