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
    'output': 'sqlite:///filtered.db',
    'remove_outliers': False,
    'mask': False,
    'kernel_size': 7,
    'nstd': 5,
    'std_thresh': 100.0,
    'deviator': 'std',
    'log': False,
}


def filter(optdict):

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

    # Filter
    network.filterData(engine_out, int(opts['kernel_size']), mask=opts['mask'],
        remove_outliers=opts['remove_outliers'], nstd=int(opts['nstd']),
        std_thresh=float(opts['std_thresh']), deviator=opts['deviator'],
        log=opts['log'])


# end of file
