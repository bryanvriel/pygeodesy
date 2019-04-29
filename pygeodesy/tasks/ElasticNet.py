#-*- coding: utf-8 -*-

import pygeodesy as pg
import pyre
from mpi4py import MPI
import sys

class ElasticNet(pg.components.task, family='pygeodesy.elasticnet'):
    """
    Fit temporal model to time series database using Elastic Net regression.
    """

    input = pyre.properties.str(default='sqlite:///data.db')
    input.doc = 'Input time series database'

    output = pyre.properties.str(default='sqlite:///sub.db')
    output.doc = 'Output time series model database'

    user = pyre.properties.str(default=None)
    user.doc = 'Python file defining temporal dictionary'

    make_new = pyre.properties.bool(default=True)
    make_new.doc = 'Flag for making a new solver instance'

    component = pyre.properties.str(default='up')
    component.doc = 'Deformation component to model'

    num_iter = pyre.properties.int(default=1)
    num_iter.doc = 'Number of ADMM iterations'

    smooth_penalty = pyre.properties.float(default=1.0)
    smooth_penalty.doc = 'Regularization parameter for L2-norm penalty'

    sparse_penalty = pyre.properties.float(default=1.0)
    sparse_penalty.doc = 'Regularization parameter for L1-norm penalty'

    correlation_length = pyre.properties.float(default=None)
    correlation_length.doc = 'Fixed characteristic length scale for station separation'

    oversmooth = pyre.properties.float(default=1.0)
    oversmooth.doc = 'Additional smoothness penalty factor'

    sigmas = pyre.properties.str(default='raw')
    sigmas.doc = 'Data sigmas for weighting from [raw, median, mean]'

    @pyre.export
    def main(self, plexus, argv):
        """
        Main entrypoint into this application.
        """
        from pyadmm.mpi.solver import ADMMSolver

        # Initialize MPI variables
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        # Create engine for input database
        engine = pg.db.Engine(url=self.input)

        # Initialize an instrument
        inst = pg.instrument.select(plexus.data_type)

        # Make a network object
        network = pg.network.Network(inst, engine, comm=comm)

        # Make or load ADMM solver
        if self.make_new:
            # Create an ADMM solver
            solver = ADMMSolver(comm=comm)
            # Initialize its data
            model = pg.network.utils.partitionData(solver, network, self, comm)
            # Set its penalties
            solver.prepareCVXOPT(self.sparse_penalty, self.smooth_penalty, DtD=False)
            # Save data
            saveData(solver, model, rank, self.component)
            
        else:
            # Load pre-computed data
            solver, model = loadData(rank, self.component, comm)
            # Update the penalties
            updatePenalties(solver, self, model, rank, self.component)

        # Solve
        m = solver.solve(N_iter=self.num_iter)

        # Master distributes the solutions to all stations
        if rank == 0:
            # Create engine for output
            engine_out = pg.db.Engine(url=self.output)
            # Distribute solutions
            distributeSolutions(m, engine_out, model, network, self)
            # Make sure to write out metadata
            engine.meta().to_sql('metadata', engine_out.engine, if_exists='replace')


# end of file
