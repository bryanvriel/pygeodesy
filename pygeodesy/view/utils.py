#-*- coding: utf-8 -*-

def model_and_detrend(data, engine, statname, component, model):

    # Get list of tables in the database
    tables = engine.tables(asarray=True)

    # Keys to look for
    model_comp = '%s_%s' % (model, component) if model != 'filt' else 'None'
    filt_comp = 'filt_' + component

    # Construct list of model components to remove (if applicable)
    if model == 'secular':
        parts_to_remove = ['seasonal', 'transient', 'step']
    elif model == 'seasonal':
        parts_to_remove = ['secular', 'transient', 'step']
    elif model == 'transient':
        parts_to_remove = ['secular', 'seasonal', 'step']
    elif model == 'step':
        parts_to_remove = ['secular', 'seasonal', 'transient']
    elif model in ['full', 'filt']:
        parts_to_remove = []
    else:
        assert False, 'Unsupported model component %s' % model

    # Make the model and detrend the data
    fit = np.nan * np.ones_like(data)
    if model_comp in tables:

        # Read full model data
        fit = pd.read_sql_table('full_' + component, engine.engine,
            columns=[statname,]).values.squeeze()

        # Remove parts we do not want
        for ftype in parts_to_remove:
            try:
                signal = pd.read_sql_table('%s_%s' % (ftype, component), engine.engine,
                                           columns=[statname,]).values.squeeze()
                fit -= signal
                data -= signal
                print('removed', ftype)
            except ValueError:
                pass

    elif filt_comp in tables and model_comp not in tables:
        fit = pd.read_sql_table('filt_%s' % component, engine.engine,
            columns=[statname,]).values.squeeze()

    return fit

# end of file
