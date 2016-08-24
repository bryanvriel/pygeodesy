#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import pandas as pd
import sys

from .Network import Network
from pygeodesy.db.Engine import Engine
import pygeodesy.instrument as instrument
from pygeodesy.model import Model

from giant.utilities import timefn
import giant.solvers as solvers

# Define the default options
defaults = {
    'input': None,
    'output': 'sqlite:///clean.db',
}


def clean(optdict):

    # Update the options
    opts = defaults.copy()
    opts.update(optdict)

    # Check if data type supports cleaning
    if opts['format'] not in ['sopac']:
        print('Format %s not supported. Exiting.' % opts['format'])
        return

    # Create engine for input database
    engine = Engine(url=opts['input'])

    # And for output
    engine_out = Engine(url=opts['output'])
    # Also initialize it
    engine_out.initdb(new=True, ref_engine=engine)

    # Initialize an instrument
    inst = instrument.select(opts['type'])

    # Make a network object
    network = Network(inst, engine)

    # Clean
    network.preprocess(engine_out)



# end of file
