#-*- coding: utf-8 -*-

from mpi4py import MPI
import sys

from pygeodesy.db.Engine import Engine
import pygeodesy.instrument as instrument
from .Network import Network
from .utils import *

# Define the default options
defaults = {
    'input': None,
    'output': 'sqlite:///enet.db',
    'nvalid': 100,
    't0': 0.0,
    'tf': 3000.0,
    'penalty': 1.0,
    'user': 'userCollection.py',
    'iter': 1,
    'make_new': True,
    'component': 'up',
    'num_iter': 20,
    'sparse_penalty': 1.0,
    'smooth_penalty': 1.0,
    'correlation_length': None,
    'oversmooth': 1.0,
    'scale': 1.0,
    'sigmas': 'raw',
}


def elasticnet(optdict):

    from pyadmm.mpi.solver import ADMMSolver

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

    # Make or load ADMM solver
    if opts['make_new']:
        # Create an ADMM solver
        solver = ADMMSolver(comm=comm)
        # Initialize its data
        model = partitionData(solver, network, opts, comm)
        # Set its penalties
        solver.prepareCVXOPT(float(opts['sparse_penalty']), 
            float(opts['smooth_penalty']), DtD=False)
        # Save data
        saveData(solver, model, rank, opts['component'])
        
    else:
        # Load pre-computed data
        solver, model = loadData(rank, opts['component'], comm)
        # Update the penalties
        updatePenalties(solver, opts, model, rank, opts['component'])

    # Solve
    m = solver.solve(N_iter=int(opts['num_iter']))

    # Master distributes the solutions to all stations
    if rank == 0:
        # Create engine for output
        engine_out = Engine(url=opts['output'])
        # Distribute solutions
        distributeSolutions(m, engine_out, model, network, opts)
        # Make sure to write out metadata
        engine.meta().to_sql('metadata', engine_out.engine, if_exists='replace')



# end of file
