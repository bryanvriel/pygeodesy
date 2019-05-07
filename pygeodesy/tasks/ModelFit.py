#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import pandas as pd
import sys

import pygeodesy as pg
import pyre

from giant.utilities import timefn

class ModelFit(pg.components.task, family='pygeodesy.modelfit'):
    """
    Fit temporal model to time series database.
    """

    input = pyre.properties.str(default='sqlite:///data.db')
    input.doc = 'Input time series database'

    output = pyre.properties.str(default='sqlite:///sub.db')
    output.doc = 'Output time series model database'

    user = pyre.properties.str()
    user.doc = 'Python file defining temporal dictionary'

    nstd = pyre.properties.int(default=5)
    nstd.doc = 'Number of deviations for outlier threshold (default: 5)'

    t0 = pyre.properties.float(default=None)
    t0.doc = 'Starting decimal year of model (default: None)'

    tf = pyre.properties.float(default=None)
    tf.doc = 'Ending decimal year of model (default: None)'

    penalty = pyre.properties.float(default=1.0)
    penalty.doc = 'Regularization parameter for model estimator (default: 1.0)'

    output_amp = pyre.properties.str(default=None)
    output_amp.doc = 'Filename for optionally saving seasonal amplitudes for each station'

    output_phase = pyre.properties.str(default=None)
    output_phase.doc = 'Filename for optionally saving seasonal phase for each station'

    num_iter = pyre.properties.int(default=1)
    num_iter.doc = 'Number of iterations for iterative least squares (default: 1)'

    rw_iter = pyre.properties.int(default=1)
    rw_iter.doc = 'Number of re-weighting iterations for LassoRegression (default: 1)'

    solver = pyre.properties.str(default='RidgeRegression')
    solver.doc = 'Name of solver from [RidgeRegression, LassoRegression]'

    scale = pyre.properties.float(default=1.0)
    scale.doc = 'Scale observations by factor (default: 1.0)'

    special_stats = pyre.properties.str(default=None)
    special_stats.doc = 'List of stations to allow for 10x higher std threshold'

    std_thresh = pyre.properties.float(default=1.0e10)
    std_thresh.doc = 'Absolute deviation threshold for bad stations (default: 1.0e10)'

    min_timespan = pyre.properties.float(default=365.0)
    min_timespan.doc = 'Minimum timespan (days) of valid data to keep station (default: 365.0)'

    min_valid = pyre.properties.int(default=100)
    min_valid.doc = 'Minimum number of valid observations to keep station (default: 100)'

    @pyre.export
    def main(self, plexus, argv):
        """
        Main entrypoint into this application.
        """
        # MPI variables
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        # Create engine for input database
        engine = pg.db.Engine(url=self.input)

        # Initialize an instrument
        inst = pg.instrument.select(plexus.data_type)

        # Make a network object
        network = pg.network.Network(inst, engine, comm=comm)

        # Partition the states
        network.partitionStations()
        proc_nstat = len(network.sub_names)

        # Master reads in the time function collection and initializes an output engine
        if rank == 0:
            collection, iCm = load_collection(network.dates, self.user)
            engine_out = pg.db.Engine(url=self.output)
            engine_out.initdb(new=True, ref_engine=engine)
        else:
            collection = iCm = None

        # Broadcast
        print('Broadcasting data')
        collection = comm.bcast(collection, root=0)
        iCm = comm.bcast(iCm, root=0)

        # Create a model for handling the time function
        model = pg.model.Model(network.dates, collection=collection, t0=self.t0, tf=self.tf)

        # Create a solver
        try:
            Solver = getattr(pg.model.solvers, self.solver)
            if self.solver == 'LassoRegression':
                solver = Solver(model.reg_indices, self.penalty, rw_iter=self.rw_iter,
                                estimate_uncertainty=False)
            else:
                solver = Solver(model.reg_indices, self.penalty, regMat=iCm)
        except AttributeError:
            print('Specified solver not supported.')
            sys.exit()
        print(solver)

        # Make list of any special stations with higher std threshold
        if self.special_stats is not None:
            special_stats = [name.strip() for name in self.special_stats.split(',')]
        else:
            special_stats = []
        
        # Loop over components
        n_iter = self.num_iter
        for comp_count, comp in enumerate(inst.components):

            if rank == 0:
                print('%s component' % comp)

            # Read subset data frame for my set of stations
            data_df = network.get(comp, network.sub_names, with_date=True)
            sigma_df = network.get('sigma_' + comp, network.sub_names, with_date=True)

            # Loop over stations
            keep_stations = []
            nstd = self.nstd
            coeff_dat = {}
            coeff_sigma_dat = {}
            for statcnt, statname in enumerate(network.sub_names):

                # Scale data
                data_df[statname] *= self.scale
                sigma_df[statname] *= self.scale

                # Get the data (by reference) for this component and station
                dat = data_df[statname].values
                wgt = 1.0 / sigma_df[statname].values

                # Initialize results dictionary for the partitions
                zero_arr = np.zeros_like(network.tdec)
                results = {'seasonal': zero_arr.copy()}

                # Iterative least squares
                isStatGood = True
                for iternum in range(n_iter):
        
                    # Remove any obvious outliers
                    if iternum > 1:
                        outlierInd = np.abs(dat) > 1000.0
                    else:
                        outlierInd = np.abs(dat) > 2000.0
                    dat[outlierInd] = np.nan

                    # Construct subset indices for inversion
                    ind = np.isfinite(dat) * np.isfinite(wgt) * model.time_mask
                    nvalid = ind.nonzero()[0].size
                    if nvalid < self.min_valid:
                        print('Skipping %s due to too few good data' % statname)
                        isStatGood = False
                        break

                    # Check time span of data matches minimum time span
                    valid_dates = network.dates[ind]
                    tspan = (valid_dates[-1] - valid_dates[0]).days
                    if tspan < self.min_timespan:
                        print('Skipping %s due to too small timespan' % statname)
                        isStatGood = False
                        break

                    # Perform least squares
                    mvec = model.invert(solver, dat) #, wgt=wgt)

                    # Save coefficients and uncertainties
                    coeff_dat[statname] = mvec
                    coeff_sigma_dat[statname] = np.sqrt(np.diag(model.Cm))

                    # Model performs reconstruction (only for detecting outliers)
                    fit_dict = model.predict(mvec)
                    filt_signal = fit_dict['full']
                    
                    # Compute misfit and standard deviation
                    misfit = dat - filt_signal
                    stdev = np.nanstd(misfit)
                    if stdev > self.std_thresh:
                        print('Skipping %s due to high stdev' % statname)
                        isStatGood = False
                        break
                    print(' - sigma: %f   %d-sigma: %f' % (stdev, nstd, nstd*stdev))

                    # Remove outliers
                    if statname in special_stats:
                        outlierInd = np.abs(misfit) > (10*stdev)
                    else:
                        outlierInd = np.abs(misfit) > (nstd*stdev)
                    dat[outlierInd] = np.nan

                if isStatGood:
                    keep_stations.append(statname)

            # Make coefficient data frames
            coeff_df = pd.DataFrame(coeff_dat)
            coeff_sigma_df = pd.DataFrame(coeff_sigma_dat)
            del coeff_dat, coeff_sigma_dat

            # Keep only the good stations
            results = []
            for df, name in ((coeff_df, 'coeff_' + comp), (coeff_sigma_df, 'coeff_sigma_' + comp)):
                sub_df = df[keep_stations]
                sub_df.name = name
                results.append(sub_df)
            Ndf = len(results)

            comm.Barrier()

            # Send my results to the master for writing
            if rank == 0:

                # Get results from each processor and add to table 
                for pid in range(1, size):

                    # Get the data
                    proc_results = comm.recv(source=pid, tag=77)
                    proc_stations = comm.recv(source=pid, tag=87)

                    # Update list of stations and seasonal data
                    keep_stations.extend(proc_stations)

                    # Update the data frames
                    for i in range(Ndf):
                        name = results[i].name
                        results[i] = results[i].join(proc_results[i])
                        results[i].name = name

                # Initialize summary dictionary
                out_dict = {'secular': {'DATE': network.dates},
                    'seasonal': {'DATE': network.dates},
                    'step': {'DATE': network.dates},
                    'transient': {'DATE': network.dates},
                    'full': {'DATE': network.dates}}

                # Compute the model fits
                coeff_df, coeff_sigma_df = results[0], results[1]
                for statname in keep_stations:
                    m = coeff_df[statname].values.squeeze()
                    fit_dict = model.predict(m, sigma=False)
                    for ftype in ('secular', 'seasonal', 'step', 'transient', 'full'):
                        out_dict[ftype][statname] = fit_dict[ftype]

                # Write results to database
                for i in range(Ndf):
                    results[i].to_sql(results[i].name, engine_out.engine, if_exists='replace')

                for ftype in ('secular', 'seasonal', 'step', 'transient', 'full'):
                    df = pd.DataFrame(out_dict[ftype])
                    df.to_sql('%s_%s' % (ftype, comp), engine_out.engine, if_exists='replace')
                    
                # Also load the metadata and keep the good stations
                meta = engine.meta()
                meta_sub = meta[np.in1d(meta['id'].values, keep_stations)].reset_index(drop=True)
                meta_sub.to_sql('metadata', engine_out.engine, if_exists='replace')

                # Write seasonal phases if requested
                seasonal_bool = ((comp == 'up') * ((self.output_phase is not None) +
                                (self.output_amp is not None)))
                if seasonal_bool:

                    raise NotImplementedError('No seasonal output support yet')

                    # Compute amplitude and phase for station
                    amp, phs = model.computeSeasonalAmpPhase()

                    if self.output_phase is not None and comp == 'up':
                        with open(self.output_phase, 'w') as pfid:
                            for statname, (amp,phs) in seasonal_dat.items():
                                stat_meta = meta_sub.loc[meta_sub['id'] == statname]
                                pfid.write('%f %f %f 0.5\n' % (float(stat_meta['lon']),
                                    float(stat_meta['lat']), phs))

                    # Write seasonal amps if requested
                    if self.output_amp is not None and comp == 'up':
                        with open(self.output_amp, 'w') as afid:
                            for statname, (amp,phs) in seasonal_dat.items():
                                stat_meta = meta_sub.loc[meta_sub['id'] == statname]
                                afid.write('%f %f %f 0.5\n' % (float(stat_meta['lon']),
                                    float(stat_meta['lat']), amp))


                # Write secular data if requested
                #if self.output_secular is not None:
                #    pass
                #    with open(self.output_secular, 'w') as sfid:
                #        for statname in secular_dat.items():
                        
            else:
                comm.send(results, dest=0, tag=77)
                comm.send(keep_stations, dest=0, tag=87)
                del data_df, sigma_df

            comm.Barrier()


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
        print(' - loading default')
        collection = loadDefaultCollection(dates)
    npar = len(collection)

    # Also try to build a prior covariance matrix
    try:
        collfun = imp.load_source('build', inputs.user)
        Cm = collfun.computeCm(collection)
        iCm = np.linalg.inv(Cm)
    except:
        iCm = None

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
