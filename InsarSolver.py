#-*- coding: utf-8 -*-

import numpy as np
from mpi4py import MPI
import pickle
import time as pytime
import h5py
import copy
import sys

import tsinsar as ts
from SparseConeQP import MintsOpt
from matutils import dmultl

np.seterr(all='ignore')

class InsarSolver:
    """
    Use mpi4py to do time series inversion in parallel.
    """

    def __init__(self, data, model, comm=None, solver_type='lsqr'):
        """
        Initiate InsarSolver class.

        Parameters
        ----------
        data: Insar
            Insar instance containing InSAR data.
        model: Model
            Model instance containing the time series model.
        comm: None or MPI.Comm, optional
            MPI communicator. Default: None
        solver_type: str {'lasso'}, optional
            Specify type of solver to use:

            ``lsqr``
                Use standard least squares w/ no regularization (Default).
            ``lasso``
                Use LASSO L1-regularization.
            ``robust``
                Use robust, L1-misfit w/ no regularization.
            ``ridge``
                Use ridge L2-regularization.
        """

        # MPI initialization
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.n_workers = self.comm.Get_size()

        # Save the data and rep objects
        self.data = data
        self.model = model

        # Save the solver type
        self.solver_type = solver_type
        if solver_type not in ['lsqr', 'lasso', 'robust', 'ridge']:
            raise NotImplementedError('Unsupported solver.')
        
        return


    def solve(self, chunks, resultDict, l2=1.0):
        """
        Invert the time series chunk by chunk.

        Parameters
        ---------
        chunks: list
            List of chunk slices.
        resultDict: dict
            Dictionary of {funcString: dataObj} pairs where funcString is a str
            in ['full', 'secular', 'seasonal', 'transient'] specifying which
            functional form to reconstruct, and dataObj is an Insar object to
            with an appropriate H5 file to store the reconstruction.
        l2: float, optional
            L2-regularization penalty.
        """

        # Instantiate a solver
        if self.solver_type == 'lasso':
            solver = MintsOpt(0.0, 0.0, 0.0, cutoff=self.cutoff, maxiter=1,
                weightingMethod=weightingMethod)
        elif self.solver_type == 'ridge':
            solver = RidgeRegression(self.model.reg_indices, l2)
        elif self.solver_type == 'lsqr':
            solver = LSQR()

        # Workers won't be storing global arrays
        if self.rank != 0:
            m_global = None

        # Initialize row data types for MPI Gathering
        par_rowtype = MPI.FLOAT.Create_contiguous(self.model.npar)
        par_rowtype.Commit()

        # Loop over the chunks
        for chunk in chunks:

            t0 = pytime.time()

            # Get the data chunks, which are now numpy arrays
            d = self.data.getChunk(chunk[0], chunk[1], dtype='igram')
            wgt = self.data.getChunk(chunk[0], chunk[1], dtype='weight')
            nifg, ny, nx = d.shape
            npix = ny * nx

            # Reshape into 2D arrays
            d = np.swapaxes(d.reshape((nifg,npix)), 0, 1)
            wgt = np.swapaxes(wgt.reshape((nifg,npix)), 0, 1)

            # Determine my portion of the data chunk
            nominal_load = npix // self.n_workers
            if self.rank == self.n_workers - 1:
                procN = npix - nominal_load * self.rank
            else:
                procN = nominal_load
            istart = nominal_load * self.rank
            iend = istart + procN
            sendcnts = self.comm.gather(procN, root=0)

            # Allocate array for local chunk results
            m = np.zeros((procN,self.model.npar), dtype=np.float32)
            # And master allocates array for global results
            if self.rank == 0:
                m_global = np.zeros((npix,self.model.npar), dtype=np.float32)

            # Loop over my portion of the data
            for cnt, i in enumerate(range(istart,iend)):
                # Get data for pixel 
                dpix = d[i,:]
                wpix = wgt[i,:]
                ind = np.isfinite(dpix)
                # Perform inversion and save parameters
                m[cnt,:], qwgt = solver.invert(self.model.G[ind,:], 
                    dpix[ind], wgt=wpix[ind])

            # Gather the final results
            self.comm.Gatherv(m, [m_global, (sendcnts, None), par_rowtype], root=0)
            # Re-shape results to match chunk dimensions
            if self.rank == 0:
                m_global = np.swapaxes(m_global, 0, 1).reshape((self.model.npar,ny,nx))

            # Model class will perform the reconstruction and write data
            self.model.predict(m_global, resultDict, chunk)
            self.comm.Barrier()
            if self.rank == 0:
                print('Finished chunk', chunk, 'in %f seconds' % (pytime.time() - t0))

        par_rowtype.Free()
        return

    
    def updateWeights(self, λ0):
        """
        Update regularization penalties using distance between stations. Here, λ0 is
        a global penalty array visible to all workers.
        """
        # Loop over my pixels
        nspl = self.npar - self.cutoff
        for i in range(self.procN):

            # Get neighbor indices and distance-dependent weights for this pixel
            nnInds = self.neighbors[i]
            nnWgts = self.neighborWeights[i]
            numNN = len(nnInds)
            penArray = np.zeros((numNN,nspl))
            wgtArray = np.zeros((numNN,nspl))

            # Fetch the neighbor penalty values and weights
            for nnInd, nnWgt, kk in zip(nnInds, nnWgts, range(numNN)):
                penArray[kk,:] = λ0[nnInd,self.cutoff:]
                wgtArray[kk,:] = nnWgt

            # Compute weighted median
            self.λ[i,self.cutoff:] = WU.computeMedian(wgtArray, penArray, self.refPenalty)

        return


class RidgeRegression:
    """
    Simple ridge regression (L2-regularization on amplitudes).
    """
    def __init__(self, reg_indices, refPenalty):
        self.reg_indices = reg_indices
        self.penalty = refPenalty
        return
    def invert(self, G, d, wgt=None):
        regMat = np.zeros((G.shape[1],G.shape[1]))
        regMat[self.reg_indices,self.reg_indices] = self.penalty
        if wgt is not None:
            GtG = np.dot(G.T, ts.dmultl(wgt**2, G)) + regMat
            Gtd = np.dot(G.T, wgt**2 * d)
        else:
            GtG = np.dot(G.T, G) + regMat
            Gtd = np.dot(G.T, d)
        m = np.linalg.lstsq(GtG, Gtd)[0]
        return m, 0.0


class LSQR:
    """
    Simple wrapper around numpy.linalg.lstsq.
    """
    def invert(self, G, d, wgt=None):
        if wgt is not None:
            m = np.linalg.lstsq(ts.dmultl(wgt, G), wgt*d)[0]
        else:
            m = np.linalg.lstsq(G, d)[0]
        return m, 0.0


# end of file
