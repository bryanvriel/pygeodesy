#-*- coding: utf-8 -*-

import pyre
import pygeodesy as pg

class CommonModeEstimation(pg.components.task, family='pygeodesy.cme'):
    """
    Estimate common mode error for a network of stations and remove from data.
    """

    input = pyre.properties.str(default='sqlite:///data.db')
    input.doc = 'Input time series database'

    output = pyre.properties.str(default='sqlite:///cme.db')
    output.doc = 'Output filtered time series database'

    remove = pyre.properties.bool(default=True)
    remove.doc = 'Flag to remove estimated common mode signal from data (default: True)'

    method = pyre.properties.str(default='pca')
    method.doc = 'Common mode error estimation method (default: pca)'

    num_components = pyre.properties.int(default=1)
    num_components.doc = 'Number of components to model common mode signal (default: 1)'

    beta = pyre.properties.float(default=1.0)
    beta.doc = 'Regularization parameter for ALS estimator (default: 1.0)'

    max_step = pyre.properties.int(default=30)
    max_step.doc = 'Number of iterations for ALS estimator (default: 30)'

    qscale = pyre.properties.float(default=1.0)
    qscale.doc = 'Scale factor for quiver plot (default: 1.0)'

    @pyre.export
    def main(self, plexus, argv):
        """
        Main entrypoint into this application.
        """
        # Create engine for input database
        engine = pg.db.Engine(url=self.input)

        # And for output
        engine_out = pg.db.Engine(url=self.output)
        # Also initialize it
        engine_out.initdb(new=True, ref_engine=engine)

        # Initialize an instrument
        inst = pg.instrument.select(plexus.data_type)

        # Make a network object
        network = pg.network.Network(inst, engine)

        # Common mode estimation
        if self.method in ['als', 'ALS']:
            network.decompose_ALS(engine_out, n_comp=self.num_components, remove=True,
                                  plot=True, beta=self.beta, max_step=self.max_step)
        else:
            network.decompose(engine_out, n_comp=self.num_components, method=self.method.lower(),
                              remove=self.remove, qscale=self.qscale)
        

# end of file
