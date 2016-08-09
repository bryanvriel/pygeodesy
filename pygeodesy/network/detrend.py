#-*- coding: utf-8 -*-

import numpy as np
from mpi4py import MPI
import sys

from .Network import Network
from pygeodesy.db.Engine import Engine
import pygeodesy.instrument as instrument
from giant.utilities import timefn

# Define the default options
defaults = {
    'input': None,
    'output': 'sqlite:///detrended.db',
    'nproc': 1,
    'cleanonly': False,
    'remove': 'secular, transient',
    'nstd': 3,
    'nvalid': 100,
    't0': 0.0,
    'tf': 3000.0,
    'penalty': 1.0,
    'output_phase': False,
    'output_amp': False,
    'user': 'userCollection.py',
    'iter': 1
}


def detrend(optdict):

    # Update the options
    opts = defaults.copy()
    opts.update(optdict)

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Create engine for input database
    engine = Engine(url=opts['input'])

    # Initialize an instrument
    inst = instrument.select(opts['type'])

    # Make a network object
    network = Network(inst, engine, comm=comm)

    # Partition the states
    network.partitionStations()

    # Master reads in the time function collection
    if rank == 0:
        collection, iCm = load_collection(network.dates, opts['user'])
    else:
        collection = iCm = None

    # Broadcast
    collection = comm.bcast(collection, root=0)
    iCm = comm.bcast(iCm, root=0)

    # Perform inversion on my subset of the stations
    for statname in network.sub_names:

        for comp in inst.components:
            print(rank, statname, comp)

    


def load_collection(dates, userfile):
    """
    Load the collection and the prior covariance matrix.
    """
    print('Loading collection')
    import imp
    try:
        collfun = imp.load_source('build', userfile)
        collection = collfun.build(dates)
    except:
        collection = loadDefaultCollection(dates)
    npar = len(collection)

    # Also try to build a prior covariance matrix
    try:
        collfun = imp.load_source('build', inputs.user)
        Cm = collfun.computeCm(collection)
        iCm = np.linalg.inv(Cm)
    except:
        iCm = np.eye(npar)

    return collection, iCm


def loadDefaultCollection(t):
    """
    Load default time function collection.
    """
    tstart, tend = t[0], t[-1]

    collection = timefn.TimefnCollection()
    poly = timefn.fnmap['poly']
    ispl = timefn.fnmap['isplineset']
    periodic = timefn.fnmap['periodic']

    collection.append(poly(tref=tstart, order=1, units='years'))
    collection.append(periodic(tref=tstart, units='weeks', period=0.5, tmin=tstart, tmax=tend))
    collection.append(periodic(tref=tstart, units='weeks', period=1.0, tmin=tstart, tmax=tend))
    for nspl in [32, 16, 8, 4]:
        collection.append(ispl(order=3, num=nspl, units='years', tmin=tstart, tmax=tend))

    return collection


# end of file
