#-*- coding: utf-8 -*-

import pyre
import pygeodesy as pg
import pandas as pd
import os

class Detrend(pg.components.task, family='pygeodesy.detrend'):
    """
    Remove time series model from time series data.
    """

    input = pyre.properties.str(default='sqlite:///data.db')
    input.doc = 'Input time series database'

    output = pyre.properties.str(default='sqlite:///detrended.db')
    output.doc = 'Output detrended time series database'

    model = pyre.properties.str()
    model.doc = 'Time series model database'

    remove = pyre.properties.str(default='secular, seasonal')
    remove.doc = 'List of model components to remove'

    scale = pyre.properties.float(default=1.0)
    scale.doc = 'Scale observations by factor'

    @pyre.export
    def main(self, plexus, argv):
        """
        Main entrypoint into this application.
        """
        # Create engine for input database
        engine = pg.db.Engine(url=self.input)

        # Create engine for model database
        model_engine = pg.db.Engine(url=self.model)

        # Create engine for detrended database
        outpath = self.output.split('///')[-1]
        if os.path.isfile(outpath):
            print('Removing old file', outpath)
            os.remove(outpath)
        engine_out = pg.db.Engine(url=self.output)
        engine_out.initdb(ref_engine=engine)

        # Initialize an instrument
        inst = pg.instrument.select(plexus.data_type)

        # Get list of model components to remove
        parts_to_remove = [s.strip() for s in self.remove.split(',')]
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
            data_df *= self.scale
            sigma_df *= self.scale

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
                print('Saving model component', model_comp)
                secular_df = pd.read_sql_table('%s_%s' % (model_comp, component),
                    model_engine.engine, index_col='DATE', columns=read_columns)
                secular_df.to_sql('%s_%s' % (model_comp, component),
                    engine_out.engine, if_exists='replace')


# end of file
