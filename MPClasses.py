#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import operator
from multiprocessing import Process, Array
from matutils import dmultl
import sys


class MPInvert(Process):


    def __init__(self, shared, subbed, G, solver, istart, cutoff):
        """
        Send data to Process parent class.
        """

        # Extract the data and save to self
        self.shared = shared
        self.bool_list, self.north, self.east, self.up = subbed[:4]
        self.wn, self.we, self.wu = subbed[4:7]
        self.penn, self.pene, self.penu = subbed[7:]
        self.G, self.solver, self.istart = G, solver, istart
        self.cutoff = cutoff
        # Init the parent Process class
        Process.__init__(self)


    def run(self):
        """
        Overload the Process.start() method.
        """
        # Cache parameters and arrays
        nstat = self.north.shape[1]
        ind = self.istart
        solver = self.solver
        cutoff = self.cutoff
        shared = self.shared

        # Check if penalties are arrays
        arrflag = [isinstance(arr, np.ndarray) for arr in [self.penn,self.pene,self.penu]]
        arrflag = reduce(operator.mul, arrflag, 1)

        # Loop over my portion of GPS stations
        for jj in xrange(nstat):
            # Unpack component-wise indices of valid observations
            bool_east, bool_north, bool_up = self.bool_list[jj]
            # Extract valid observations
            dnorth, deast, dup = (self.north[bool_north,jj], 
                                  self.east[bool_east,jj], 
                                  self.up[bool_up,jj])
            wn, we, wu = (self.wn[bool_north,jj], 
                          self.we[bool_east,jj], 
                          self.wu[bool_up,jj])
            Gn, Ge, Gu = self.G[bool_north,:], self.G[bool_east,:], self.G[bool_up,:]
            # Perform estimation and store weights
            if arrflag:
                northPen, eastPen, upPen = self.penn[jj,:], self.pene[jj,:], self.penu[jj,:]
            else:
                northPen, eastPen, upPen = self.penn, self.pene, self.penu
            shared.m_north[:,ind], qn = solver.invert(dmultl(wn,Gn), wn*dnorth, northPen)
            shared.m_east[:,ind], qe  = solver.invert(dmultl(we,Ge), we*deast, eastPen)
            shared.m_up[:,ind], qu    = solver.invert(dmultl(wu,Gu), wu*dup, upPen)
            # Now modify the shared penalty array
            if arrflag:
                shared.penn[ind,:] = qn[cutoff:]
                shared.pene[ind,:] = qe[cutoff:]
                shared.penu[ind,:] = qu[cutoff:]
            ind += 1

        # done
        return


class MPWeights(Process):
    """
    A multiprocessing class to perform weighted mean/median for multiple GPS stations in parallel.
    Uses the multiprocess.Process parent class.
    """


    def __init__(self, allWeights, shared, penalty, penn, pene, penu, istart, wtype='mean',
                 combineHorizontal=False):

        self.allWeights, self.shared = allWeights, shared
        self.penn, self.pene, self.penu = penn, pene, penu # shared memory
        self.istart, self.wtype = istart, wtype
        if type(penalty) is dict:
            self.npenalty = penalty['north']
            self.epenalty = penalty['east']
            self.upenalty = penalty['up']
        else:
            self.npenalty = self.epenalty = self.upenalty = penalty 
        self.combineHorizontal = combineHorizontal
        Process.__init__(self)


    def run(self):
        """
        Overload the start() method.
        """
        # Cache paremeters and arrays
        allWeights = self.allWeights
        spenn, spene, spenu = self.shared.penn, self.shared.pene, self.shared.penu
        ind = self.istart
        nstat = allWeights.shape[0]
        npar = self.penn.shape[1]
        combineHorizontal = self.combineHorizontal
        
        # Choose my estimator
        if 'mean' in self.wtype:
            estimator = self.computeMean
        elif 'median' in self.wtype:
            estimator = self.computeMedian
        else:
            assert False, 'unsupported weight type. must be mean or median'
        
        # Loop over my portion of GPS stations
        for jj in xrange(nstat):
            # Extract weights
            #plt.semilogy(spenn[ind,:], 'b')
            #plt.semilogy(spene[ind,:], 'r')
            #plt.savefig('figures/penalties_%03d.png' % ind)
            #plt.clf()
            weights = np.tile(np.expand_dims(allWeights[jj,:], axis=1), (1,npar))
            # Compute weighted penalty
            if combineHorizontal:
                wgtPen = estimator(weights, spenn+spene, 0.5*(self.npenalty+self.epenalty))
                self.penn[ind,:] = wgtPen
                self.pene[ind,:] = wgtPen
            else:
                self.penn[ind,:] = estimator(weights, spenn, self.npenalty)
                self.pene[ind,:] = estimator(weights, spene, self.epenalty)
            self.penu[ind,:] = estimator(weights, spenu, self.upenalty)
            ind += 1

        return


    @staticmethod
    def computeMean(weights, dataArray, scale):
        """
        Compute weighted mean.
        """
        total_weight = np.sum(weights, axis=0) # (1,npar)
        weighted_sum = np.sum(weights * dataArray, axis=0)
        return scale * weighted_sum / total_weight


    @staticmethod
    def computeMedian(weights, dataArray, scale):
        """
        Compute weighted median.
        """
        assert dataArray.shape == weights.shape, 'mismatched shapes for weighted array'
        total_weight = np.sum(weights, axis=0)   # (1,npar)
        indsort = np.argsort(dataArray, axis=0)

        # Loop over parameters and find median for each
        wmedians = np.zeros((dataArray.shape[1],))
        for jj in xrange(dataArray.shape[1]):
            # Get the sorting indices
            ind = indsort[:,jj]
            # Sort weights and data
            sorted_weights = weights[ind,jj]
            sorted_data = dataArray[ind,jj]
            # Compute weighted median
            sum = total_weight[jj] - sorted_weights[0]
            threshold = 0.5 * total_weight[jj]
            k = 0
            while sum > threshold:
                k += 1
                sum -= sorted_weights[k]
            wmedians[jj] = sorted_data[k]

        return scale * wmedians
            


def makeSharedArrays(shapeList):
    """
    Loops through a sequence of shapes and generates a shared memory Array for each shape.
    """
    out_arrays = []
    for shape in shapeList:
        nx,ny = shape
        arr = Array('d', nx*ny)
        out_arrays.append(np.frombuffer(arr.get_obj()).reshape(shape))

    return out_arrays
