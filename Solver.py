#!-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from SparseConeQP import BaseOpt

class Solver:
    """
    Abstract class for performing time series inverisons for various data types.
    """

    def __init__(self, dataObj, timeRep, solver, seasrep=None):
        """
        Initialize with TimeSeries and TimeRepresentation objects.
        """
        self.data = dataObj
        self.timeRep = timeRep
        self.components = dataObj.components
        self.ncomp = dataObj.ncomp
        
        # Get the time rep matrices
        #self.Gss, self.Gtran = self.timeRep.matrixNoReg, self.timeRep.matrixReg
        self.G = self.timeRep.matrix
        self.cutoff = self.timeRep.cutoff

        # Store the problem sizes
        self.ndat = self.G.shape[0]
        self.nstat = self.data.nstat
        self.npar = timeRep.npar

        # Specify the solver class
        if solver == 'cvxopt':
            self.solverClass = BaseOpt
        elif solver == 'tnipm':
            try:
                from tnipm import TNIPM
                self.solverClass = TNIPM
            except ImportError:
                self.solverClass = BaseOpt
        else:
            assert False, 'unsupported solver type'

        return

    
    def invert(self, **kwargs):
        raise NotImplementedError('Child classes must define an invert method')

    
    def viewIteration(self, statname, shared, east_pen, north_pen, G, tdec, itcnt):

        raise NotImplementedError('Need to modify for mpi4py')
        fig = plt.figure(figsize=(16,8))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        axes_pairs = [(ax1, ax3), (ax2, ax4)]

        # Compute current transient signal
        statind = (self.data.name == statname).nonzero()[0]
        cnt = 0
        for component, penarray in [('east', east_pen), ('north', north_pen)]:

            m = getattr(shared, 'm_' + component)[:,statind].squeeze()
            pen = penarray[statind,:].squeeze()
            npar = pen.size
            fitc = np.dot(G[:,:self.cutoff], m[:self.cutoff])
            fitt = np.dot(G[:,self.cutoff:], m[self.cutoff:])
            dat = getattr(self.statDict[statname], component)

            ax_top, ax_bottom = axes_pairs[cnt]

            ax_top.plot(tdec, dat - fitc, 'o', alpha=0.6)
            ax_top.plot(tdec, fitt, '-r', linewidth=3)
            ax_top.set_xlim([tdec[0], tdec[-1]])
            ax_bottom.plot(np.arange(npar), pen)
            ax_bottom.set_xlim([0, npar])
            cnt += 1

        plt.suptitle(statname, y=0.94, fontsize=18)
        #plt.show(); assert False
        fig.savefig('figures/%s_iterations_%03d.png' % (statname, itcnt), bbox_inches='tight')
        plt.close(fig)
        return


    def xval(self, kfolds, lamvec, rep, cutoff, tdec, maxiter=1, statlist=None, plot=False):
        """
        Performs k-fold cross validation on each component of the data object.
        """

        if statlist is None:
            statlist = self.data.name

        # Make representation and cutoff dictionaries if not provided
        if type(rep) is not dict:
            repDict = {}
            for statname in statlist:
                repDict[statname] = rep
        else:
            repDict = rep

        if type(cutoff) is not dict:
            cutoffDict = {}
            for statname in statlist:
                cutoffDict[statname] = cutoff
        else:
            cutoffDict = cutoff

        if plot:
            if not os.path.exists('figures'):
                os.mkdir('figures')

        for statname in statlist:

            stat = self.statDict[statname]
            print(' - cross validating at station', statname)

            # Construct dictionary matrix
            Gref = np.asarray(ts.Timefn(repDict[statname], tdec-tdec[0])[0], order='C')
            N,Npar = Gref.shape
            cutoff = cutoffDict[statname]

            # Sparse solver
            l1 = sp.BaseOpt(cutoff=cutoff, maxiter=maxiter)

            if plot:
                fig, (axes) = plt.subplots(nrows=self.data.ncomp, ncols=1)
                if self.data.ncomp == 1:
                    axes = [axes]

            # Loop over components
            for ind in range(self.ncomp):
    
                # Get valid data indices
                component = self.data.components[ind]
                dat = getattr(stat, component)
                wgt = getattr(stat, 'w_' + component)
                ind = np.isfinite(dat) * np.isfinite(wgt)
                dat = dat[ind]
                wgt = wgt[ind]
                G = Gref[ind,:].copy()

                # Do xval
                penalty, error = l1.xval(kfolds, lamvec, dmultl(wgt,G), wgt*dat)
                print('     - finished %s with optimal penalty of %f' % (component, penalty))

                if plot:
                    axes[ind].semilogx(lamvec, error)

            if plot:
                fig.savefig()
                
        
# end of file
