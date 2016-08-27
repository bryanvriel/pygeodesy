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
    'output': 'sqlite:///cme.db',
    'remove': True,
    'method': 'pca',
    'num_components': 1,
}


def cme(optdict):

    # Update the options
    opts = defaults.copy()
    opts.update(optdict)

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

    # Common mode estimation
    network.decompose(engine_out, n_comp=int(opts['num_components']), 
        method=opts['method'].lower(), remove=opts['remove'])
        

# end of file
