#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
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
    'model': None,
    'output': 'sqlite:///detrended.db',
    'remove': 'secular, seasonal',
    'scale': 1.0,
}

def detrend(optdict):

    # Update the options
    opts = defaults.copy()
    opts.update(optdict)

    # Create engine for input database
    engine = Engine(url=opts['input'])

    # Create engine for model database
    model_engine = Engine(url=opts['model'])

    # Create engine for detrended database
    engine_out = Engine(url=opts['output'])
    engine_out.initdb(ref_engine=engine)

    # Initialize an instrument
    inst = instrument.select(opts['type'])

    # Get list of model components to remove
    parts_to_remove = [s.strip() for s in opts['remove'].split(',')]
    print('Removing model components:', parts_to_remove)

    # Transfer metadata
    meta = model_engine.meta()
    meta.to_sql('metadata', engine_out.engine, if_exists='replace')
    statnames = meta['id'].values.tolist()

    # Columns to extract from input data frame
    read_columns = ['DATE'] + statnames

    # Loop over the components
    for component in inst.components:

        # Get a copy of the original data
        data_df = pd.read_sql_table(component, engine.engine, index_col='DATE', 
            columns=read_columns)
        sigma_df = pd.read_sql_table('sigma_' + component, engine.engine,
            index_col='DATE', columns=read_columns)

        # Scale data and sigma
        data_df *= float(opts['scale'])
        sigma_df *= float(opts['scale'])

        # Read model data by model type and remove
        model_fit = pd.read_sql_table('full_%s' % component, model_engine.engine,
            index_col='DATE')
        for ftype in parts_to_remove:
            model_df = pd.read_sql_table('%s_%s' % (ftype, component), 
                model_engine.engine, index_col='DATE')
            data_df -= model_df
            model_fit -= model_df

        # Save data
        data_df.to_sql(component, engine_out.engine, if_exists='replace')
        model_fit.to_sql('full_%s' % component, engine_out.engine, if_exists='replace')
        sigma_df.to_sql('sigma_' + component, engine_out.engine, if_exists='replace')

        for model_comp in ('secular', 'seasonal', 'transient', 'step'):
            if model_comp in parts_to_remove:
                continue
            secular_df = pd.read_sql_table('%s_%s' % (model_comp, component),
                model_engine.engine, index_col='DATE', columns=read_columns)
            secular_df.to_sql('%s_%s' % (model_comp, component),
                engine_out.engine, if_exists='replace')


# end of file
