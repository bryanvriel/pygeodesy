#-*- coding: utf-8 -*-

import numpy as np
from mpi4py import MPI
import pickle
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

    def __init__(self, data, rep, comm=None, solver_type='lsqr'):
        """
        Initiate InsarSolver class.

        Parameters
        ----------
        data: Insar
            Insar instance containing InSAR data.
        rep: TimeRepresentation
            TimeRepresentation instance.
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
        self.rep = rep

        # Construct a G matrix for inversion
        self.H = rep.matrix
        self.G = np.dot(data.Jmat, self.H)
        self.npar = self.H.shape[1]
        self.nrecon = self.H.shape[0]

        # Save the solver type
        self.solver_type = solver_type
        if solver_type not in ['lsqr', 'lasso', 'robust', 'ridge']:
            raise NotImplementedError('Unsupported solver.')
        
        return


    def solve(self, chunks, result):
        """
        Invert the time series chunk by chunk.

        Parameters
        ---------
        chunks: list
            List of chunk slices.
        result: Insar
            Insar object to store output.
        """

        # Instantiate a solver
        if self.solver_type == 'lasso':
            solver = MintsOpt(0.0, 0.0, 0.0, cutoff=self.cutoff, maxiter=1,
                weightingMethod=weightingMethod)
        elif self.solver_type == 'lsqr':
            solver = LSQR()

        # Workers won't be storing global arrays
        if self.rank != 0:
            fit_global = None

        # Initialize a row data type for MPI Gathering
        row_type = MPI.FLOAT.Create_contiguous(self.nrecon)
        row_type.Commit()

        # Loop over the chunks
        for chunk in chunks:

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
            fit = np.zeros((procN,self.nrecon), dtype=np.float32)

            # Loop over my portion of the data
            for cnt, i in enumerate(range(istart,iend)):
                # Get data for pixel 
                dpix = d[i,:]
                wpix = wgt[i,:]
                ind = np.isfinite(dpix)
                # Perform inversion
                m = solver.invert(self.G[ind,:], dpix[ind], wgt=wpix[ind])
                # Save reconstruction
                fit[cnt,:] = np.dot(self.H, m)

            # Gather the final results
            if self.rank == 0:
                fit_global = np.zeros((npix,self.nrecon), dtype=np.float32)
            self.comm.Gatherv(fit, [fit_global, (sendcnts, None), row_type], root=0)

            # Re-shape it to match chunk dimensions
            if self.rank == 0:
                fit_global = np.swapaxes(fit_global, 0, 1).reshape((self.nrecon,ny,nx))

            # Save the results
            result.setChunk(fit_global, chunk[0], chunk[1], dtype='recon', verbose=True)

        row_type.Free()        
        return

    
    def reconstruct(self, H):
        """
        Reconstruct time series for every pixel.
        """
        # Loop over my pixels
        self.cutoff = 0
        for jj in range(self.procN):
            self.recons[jj,:] = np.dot(H[:,self.cutoff:], self.m[jj,self.cutoff:])

        # Master gathers the reconstruction from all workers
        self.comm.Gatherv(self.recons, [self.recons0, (self.sendcnts, None),
                          self.recons_rowtype], root=0)
        self.recons_rowtype.Free()

        # Put the reconstruction back into the original geometry
        if self.rank == 0:
            npixel_ref = len(self.line_status)
            recons0 = np.zeros((npixel_ref,self.ntims), dtype=self.recons0.dtype)
            m0 = np.zeros((npixel_ref,self.npar), dtype=self.m0.dtype)

            recons0[self.line_status,:] = self.recons0
            m0[self.line_status,:] = self.m0
            self.recons0 = recons0
            self.m0 = m0

        return


    def invertModulated(self, insar_wgts, seas_rep, tdec, nsplmod, maxiter=1, 
        weightingMethod='log', norm_tol=1.0e-4):

        # Instantiate a solver
        if self.solver_type == 'sparse':
            objSolver = self.solverClass(0.0, 0.0, 0.0, cutoff=self.cutoff, maxiter=1,
                weightingMethod=weightingMethod)
        elif self.solver_type == 'simple':
            objSolver = self.solverClass(self.refPenalty)
        else:
            assert False

        # Reference seasonal template dictionary
        H_seas = ts.Timefn(seas_rep, tdec-tdec[0])[0]

        # Make an amplitude modulator
        rep = [['ISPLINE',[3],[nsplmod]]]
        H_mod_ref = ts.Timefn(rep, tdec-tdec[0])[0]

        # Iterate
        for ii in range(maxiter):

            if self.rank == 0: print('At iteration', ii)

            # Loop over my portion of the data
            for jj in range(self.procN):

                # Get boolean indices of valid observations and weights
                dat = self.datArr[jj,:].copy()
                ind = (np.abs(dat) < 1.0e6) * (np.abs(dat) > 1.0e-5)
                nvalid = ind.nonzero()[0].size
                if nvalid < 20:
                    self.m[jj,:] = 1.0e8
                    continue

                # Form modulated seasonal + transient dictionary
                template = np.dot(H_seas, self.m_seas[jj,:])
                mean_spline = np.mean(template)
                fit_norm = template - mean_spline
                template = fit_norm / (0.5*(fit_norm.max() - fit_norm.min()))
                H_mod = H_mod_ref.copy()
                for nn in range(nsplmod):
                    H_mod[:,nn] *= template
                G = np.column_stack((np.dot(self.Jmat, H_mod), self.G))

                # Perform inversion
                Gsub, dat, wgt = G[ind,:], dat[ind], insar_wgts[ind]
                λ = self.λ[jj,self.cutoff:].copy()
                self.m[jj,:], q = objSolver.invert(dmultl(wgt,Gsub), wgt*dat, λ)
                self.λ[jj,self.cutoff:] = q

            # Master gathers the results for this iteration
            self.comm.Gatherv(self.m, [self.m0, (self.sendcnts, None), self.par_rowtype], root=0)
            self.comm.Gatherv(self.λ, [self.λ0, (self.sendcnts, None), self.par_rowtype], root=0)

            # Broadcast the global penalties so every worker has a copy
            self.comm.Bcast([self.λ0, MPI.DOUBLE], root=0)

            # Compute weights and re-gather
            if self.solver_type == 'sparse':
                if self.rank == 0: print(' - updating weights')
                self.updateWeights(self.λ0)

        # Remove any invalid values
        if self.rank == 0:
            self.m0[self.m0 > 1.0e6] = np.nan

        self.par_rowtype.Free()        
        return

    
    def reconstructModulated(self, H_ref, seas_rep, tdec, nsplmod):
        """
        Reconstruct time series for every pixel.
        """

        # Reference seasonal template dictionary
        H_seas = ts.Timefn(seas_rep, tdec-tdec[0])[0]

        # Make an amplitude modulator
        rep = [['ISPLINE',[3],[nsplmod]]]
        H_mod_ref = ts.Timefn(rep, tdec-tdec[0])[0]

        # Loop over my pixels
        self.cutoff = 0
        for jj in range(self.procN):

            # Form modulated seasonal + transient dictionary
            template = np.dot(H_seas, self.m_seas[jj,:])
            mean_spline = np.mean(template)
            fit_norm = template - mean_spline
            template = fit_norm / (0.5*(fit_norm.max() - fit_norm.min()))
            H_mod = H_mod_ref.copy()
            for nn in range(nsplmod):
                H_mod[:,nn] *= template
            H = np.column_stack((H_mod, H_ref))

            # Mulitply
            self.recons[jj,:] = np.dot(H, self.m[jj,:])

        # Master gathers the reconstruction from all workers
        self.comm.Gatherv(self.recons, [self.recons0, (self.sendcnts, None),
                          self.recons_rowtype], root=0)
        self.recons_rowtype.Free()

        # Put the reconstruction back into the original geometry
        if self.rank == 0:
            npixel_ref = len(self.line_status)
            recons0 = np.zeros((npixel_ref,self.ntims), dtype=self.recons0.dtype)
            m0 = np.zeros((npixel_ref,self.npar), dtype=self.m0.dtype)

            recons0[self.line_status,:] = self.recons0
            m0[self.line_status,:] = self.m0
            self.recons0 = recons0
            self.m0 = m0

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
    def __init__(self, refPenalty):
        self.penalty = refPenalty
        return
    def invert(self, G, d, dummy):
        reg = np.eye(G.shape[1])
        #reg[range(18,20),range(18,20)] = 0.0
        reg[range(9),range(9)] = 0.0
        GtG = np.dot(G.T, G) + self.penalty * reg
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
        return m


# end of file
