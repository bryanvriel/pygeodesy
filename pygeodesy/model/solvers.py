#-*- coding: utf-8 -*-

"""
Class definitions for two different sparse-regularization solvers. Uses CVXOPT for convex 
optimization.

.. author:
    Bryan Riel <briel@caltech.edu>

.. dependencies:
    numpy, scipy, cvxopt
"""

import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, spdiag, mul, div, sqrt, normal, setseed, log
from cvxopt import blas, solvers, sparse, spmatrix
from sklearn.model_selection import KFold
from collections import defaultdict
import sys


class LinearRegression:
    """
    Base class for all linear regression solvers. Implements a simple linear
    least squares.
    """

    def __init__(self):
        """
        Initialize the LinearRegression class.
        """
        pass


    def invert(self, G, d, wgt=None):
        """
        Simple wrapper around numpy.linalg.lstsq.

        Parameters
        ----------
        G: (M,N) np.ndarray
            Input design matrix.
        d: (M,) np.ndarray
            Input data.
        wgt: (M,) np.ndarray, optional
            Optional weights for the data.

        Returns
        -------
        m: (N,) np.ndarray
            Output parameter vector.
        m_wgt: (N,) np.ndarray, optional
            Weights for parameters.
        """
        # Perform inversion 
        if wgt is not None:
            GtG = np.dot(G.T, dmultl(wgt**2, G))
            Gtd = np.dot(G.T, wgt**2 * d)
        else:
            GtG = np.dot(G.T, G)
            Gtd = np.dot(G.T, d)
        iGtG = np.linalg.pinv(GtG, rcond=1.0e-8)
        m = np.dot(iGtG, Gtd)
        return m, iGtG


class RidgeRegression(LinearRegression):
    """
    Simple ridge regression (L2-regularization on amplitudes).
    """

    def __init__(self, reg_indices, penalty, regMat=None, **kwargs):
        """
        Initialize the RidgeRegression class and store regularization indices
        and regularization parameter.

        Parameters
        ----------
        reg_indices: np.ndarray
            Regularization indices.
        penalty: float
            Regularization parameter.
        """
        super().__init__()
        self.reg_indices = reg_indices
        self.penalty = penalty
        self.regMat = regMat

        return

    def invert(self, G, d, wgt=None):
        """
        Perform inversion.
        Simple wrapper around numpy.linalg.lstsq.

        Parameters
        ----------
        G: (M,N) np.ndarray
            Input design matrix.
        d: (M,) np.ndarray
            Input data.
        wgt: (M,) np.ndarray, optional
            Optional weights for the data.

        Returns
        -------
        m: (N,) np.ndarray
            Output parameter vector.
        m_wgt: (N,) np.ndarray, optional
            Weights for parameters.
        """

        # Cache the regularization matrix or compute it
        regMat = self.regMat
        if regMat is None:
            regMat = np.zeros((G.shape[1], G.shape[1]))
            regMat[self.reg_indices,self.reg_indices] = self.penalty

        # Perform inversion 
        if wgt is not None:
            GtG = np.dot(G.T, dmultl(wgt**2, G))
            Gtd = np.dot(G.T, wgt**2 * d)
        else:
            GtG = np.dot(G.T, G)
            Gtd = np.dot(G.T, d)
        iGtG = np.linalg.pinv(GtG + regMat, rcond=1.0e-8)
        m = np.dot(iGtG, Gtd)
        return m, iGtG

    def __repr__(self):
        msg  = 'Ridge Regression'
        return msg



class LassoRegression(LinearRegression):
    """
    Linear regression with an L1-norm regularization function.
    """

    def __init__(self, reg_indices, penalty, reweightingMethod='log', rw_iter=1,
                 estimate_uncertainty=False, **kwargs):
        """
        Initialize the LassoRegression class and store regularization indices,
        regularization parameter, and re-weighting method.

        Parameters
        ----------
        reg_indices: np.ndarray
            Regularization indices.
        penalty: float
            Regularization parameter.
        reweightingMethod: str, {'log', 'inverse', 'isquare'}, optional
            Specify the reweighting method. Default: log.
        rw_iter: int, optional
            Number of re-weighting operations. Default: 5.
        """
        super().__init__()
        self.reg_indices = np.array(reg_indices)
        self.penalty = penalty
        self.eps = 1.0e-5
        self.cutoff = self.reg_indices[0]
        self.rw_iter = rw_iter
        self.estimate_uncertainty = estimate_uncertainty
        self.solver = SparseOpt(
            cutoff=self.cutoff, maxiter=rw_iter, eps=1.0e-4, weightingMethod=reweightingMethod
        )

    def invert(self, G, d, wgt=None):
        """
        Perform inversion.

        Parameters
        ----------
        A: (M,N) np.ndarray
            Input design matrix.
        b: (M,) np.ndarray
            Input data.
        wgt: (M,) np.ndarray, optional
            Optional weights for the data.

        Returns
        -------
        m: (N,) np.ndarray
            Output parameter vector.
        m_wgt: (N,) np.ndarray, optional
            Weights for parameters.
        """
        # Pre-multiply by wgt array
        if wgt is not None:
            A = dmultl(wgt, G)
            b = wgt * d
        else:
            A = G
            b = d
        m, n = G.shape
         
        # Call sparse solver
        x = self.solver.invert(A, b, self.penalty)[0]

        # Estimate uncertainty
        if self.estimate_uncertainty:
            # Get indices for variance reduction
            best_ind = self._selectBestBasis(np.array(A), x, d)
            Gsub = np.array(A)[:,best_ind]
            nsub = Gsub.shape[1]
            # Compute new linear algebra arrays
            if wgt is not None:
                GtG = np.dot(Gsub.T, dmultl(wgt**2, Gsub))
                Gtd = np.dot(Gsub.T, wgt**2 * d)
            else:
                GtG = np.dot(Gsub.T, Gsub)
                Gtd = np.dot(Gsub.T, d)
            # Do sub-set least squares
            iGtG = np.linalg.pinv(GtG + 0.01*np.eye(nsub), rcond=1.0e-8)
            m = np.dot(iGtG, Gtd)
            # Place in original locations
            x = np.zeros(n)
            Cm = np.zeros((n,n))
            x[best_ind] = m
            Cm[best_ind,best_ind] = np.diag(iGtG)
        else:
            Cm = np.eye(x.size)

        return x, Cm

    def _selectBestBasis(self, G, m, d, normalize=False, varThresh=0.95):
        """
        Given a sparse solution, this routine chooses the elements that give the most variance 
        reduction. 
        """
        # Cache indices
        all_ind = np.arange(len(m), dtype=self.reg_indices.dtype)
        steady_ind = np.setxor1d(all_ind, self.reg_indices)
        reg_ind = self.reg_indices

        # First remove the steady-state terms
        refVariance = np.std(d)**2
        dat = d - np.dot(G[:,steady_ind], m[steady_ind])
        variance = np.std(dat)**2
        varianceReduction = 1.0 - variance / refVariance

        # Sort the transient components of m from highest to lowest
        sortIndices = np.argsort(np.abs(m[reg_ind]))[::-1]
        sortIndices = reg_ind[sortIndices]

        # Loop over components and compute variance reduction
        bestIndices = steady_ind.tolist()
        cnt = 0
        ref_var_reduction = varianceReduction
        delta_reduction = 100.0
        while varianceReduction < varThresh:

            # Get the model fit for this component
            index = sortIndices[cnt]
            fit = np.dot(G[:,index], m[index])

            # Remove from data
            dat -= fit
            variance = np.std(dat)**2
            varianceReduction = 1.0 - variance / refVariance

            # Check if we're not getting any better
            delta_reduction = varianceReduction - ref_var_reduction
            if delta_reduction < 1.0e-6:
                break
            ref_var_reduction = varianceReduction

            bestIndices.append(index)
            cnt += 1

        return bestIndices

    def __repr__(self):
        msg  = 'Lasso Regression:\n'
        msg += '  - cutoff: %d\n' % self.cutoff
        msg += '  - penalty: %f\n' % self.penalty
        msg += '  - rw_iter: %d\n' % self.rw_iter
        return msg


class SparseOpt:
    """
    Base optimization class for monotonic time-series, i.e. for GPS-type data.
    Successive calls are made to CVXOPT for re-weighting operations. K-fold
    cross-validation is implemented.
    """

    def __init__(self, cutoff=0, maxiter=1, eps=1.0e-3, weightingMethod='log'):
        """
        Minimizes the cost function:

        J(m) = ||A*x - b||_2^2 + lambda * ||F*x||_1

        where the subscripts denote L2- or L1-norm. F is a diagonal matrix that penalizes the 
        amplitude of the elements in m. Casts the L1-regularized cost function as a second-order 
        cone quadratic problem and solves the problem using CVXOPT. Iterative re-weighting is 
        performed to update the diagonals of F.

        Arguments:
        cutoff                  number of parameters which we DO NOT regularize in inversion. 
                                Matrix A must be structured such that these parameters are the
                                first columns of A (or parameters of x)
        maxiter                 maximum number of re-weighting iterations
        """
        self.cutoff = cutoff
        self.maxiter = maxiter
        self.eps = eps
        if 'log' in weightingMethod:
            self.weightingFunc = self.logWeighting
        elif 'inverse' in weightingMethod:
            self.weightingFunc = self.inverseWeighting
        elif 'isquare' in weightingMethod:
            self.weightingFunc = self.inverseSquareWeighting
        else:
            assert False, 'unsupported weighting method'

    def invert(self, Ain, bin, penalty, pconst=0.0, sparseP=False,
               positive=False):
        """
        Calls CVXOPT Cone quadratic program solver.

        Arguments:
        Ain                     input design array of size (M x N)
        bin                     input data array of size (M)
        penalty                 floating point penalty parameter (lambda)
        eps                     small number to provide stability for re-weighting.
        positive                Flag for specifying non-negative solutions.

        Returns:
        x                       regularized least-squares solution of size (N)
        q                       array of weights used in regularization (diagonals of F)
        """

        solvers.options['show_progress'] = False
        arrflag = isinstance(penalty, np.ndarray)
        weightingFunc = self.weightingFunc

        # Convert Numpy arrays to CVXOPT matrices
        b = matrix(bin.tolist())
        A = matrix(Ain.T.tolist())
        m,n = A.size
        nspl = n - self.cutoff

        # Fill q (will modify for re-weighting)
        q = matrix(0.0, (2*n,1))
        q[:n] = -A.T * b
        q[n+self.cutoff:] = penalty

        # Fill h
        if positive:
            h = matrix(0.0, (3*n,1))
        else:
            h = matrix(0.0, (2*n,1))

        # Fill P
        P = matrix(0.0, (2*n,2*n))
        P[:n,:n] = A.T * A
        P[list(range(n)),list(range(n))] += pconst
        if sparseP:
            P = sparse(P)

        # Fill G
        if positive:
            G = matrix(0.0, (3*n,2*n))
        else:
            G = matrix(0.0, (2*n,2*n))
        eye = spmatrix(1.0, range(n), range(n))
        G[:n,:n] = eye
        G[:n,n:] = -1.0 * eye
        G[n:2*n,:n] = -1.0 * eye
        G[n:2*n,n:] = -1.0 * eye
        if positive:
            n_con = n - self.cutoff
            G[2*n+self.cutoff:,self.cutoff:n] = spmatrix(-1.0, range(n_con), range(n_con))
        G = sparse(G)

        # Perform re-weighting by calling solvers.coneqp()
        for iters in range(self.maxiter):
            soln = solvers.coneqp(P, q, G=G, h=h)
            status, x = soln['status'], soln['x'][:n]
            if status != 'optimal':
                x = np.nan * np.ones((n,))
                break
            xspl = x[self.cutoff:]
            wnew = weightingFunc(xspl)
            if arrflag: # if outputting array, use only 1 re-weight iteration
                q[n+self.cutoff:] = wnew
            else:
                q[n+self.cutoff:] = penalty * wnew
        
        return np.array(x).squeeze(), np.array(q[n:]).squeeze()


    def logWeighting(self, x):
        """
        Log re-weighting function used in sparse optimization.
        """
        ncoeff = x.size[0]
        return log(div(blas.asum(x) + ncoeff*self.eps, abs(x) + self.eps))


    def inverseWeighting(self, x):
        """
        Re-weighting function used in Candes, et al. (2009).
        """
        return div(1.0, abs(x) + self.eps)


    def inverseSquareWeighting(self, x):
        """
        Similar to Candes re-weighting but using inverse square relationship.
        """
        return div(1.0, x**2 + self.eps**2)


    def xval(self, kfolds, lamvec, A, b, random_state=None):
        """
        Define K-fold cross-validation scheme. Can choose to define training
        and testing sets using the aquisition dates ('sar') or the actual
        interferogram dates ('ifg').

        Arguments:
        kfolds                      number of folds to perform
        lamvec                      array of penalty parameters to test
        Ain                         input design array of size (M x N)
        bin                         input data of size (M)

        Returns:
        lam_min                     penalty corresponding to lowest mean square error
        error                       array of shape (lamvec.shape) containing mean square error for
                                    each penalty in lamvec.
        """

        # Separate the indices into testing and training subsets
        n, npar = A.shape
        kf = KFold(n, n_folds=kfolds, shuffle=True, random_state=random_state)

        # Loop over k-folds
        err = np.zeros((kfolds, lamvec.size), dtype=float)
        nlam = lamvec.size
        ii = 0
        for itrain, itest in kf:

            # Grab data based on indices
            Atrain = A[itrain,:]
            btrain = b[itrain]
            Atest  = A[itest,:]
            btest  = b[itest]

            # Loop over regularization parameters
            for jj in range(nlam):
                penalty = lamvec[jj]
                x = self.invert(Atrain, btrain, penalty)[0]
                misfit = btest - np.dot(Atest, x)
                err[ii,jj] = np.dot(misfit, misfit) 

            ii += 1

        # Collapse error vector by summing over all k-fold experiments
        toterr = np.sum(err, axis=0)

        # Find lambda that minimizes the error
        ind = np.argmin(toterr)
        return lamvec[ind], toterr  


####################Fast matrix utilities########################
def dmultl(dvec, mat):
    '''Left multiply with a diagonal matrix. Faster.
    
    Args:
        
        * dvec    -> Diagonal matrix represented as a vector
        * mat     -> Matrix
        
    Returns:
    
        * res    -> dot (diag(dvec), mat)'''

    res = (dvec*mat.T).T
    return res

def dmultr(mat, dvec):
    '''Right multiply with a diagonal matrix. Faster.
    
    Args:
        
        * dvec    -> Diagonal matrix represented as a vector
        * mat     -> Matrix
        
    Returns:
    
        * res     -> dot(mat, diag(dvec))'''

    res = dvec*mat
    return res 
