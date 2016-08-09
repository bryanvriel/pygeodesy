#-*- coding: utf-8 -*-

import numpy as np
from mpi4py import MPI
import pandas as pd
import sys

from .Network import Network
from pygeodesy.db.Engine import Engine
import pygeodesy.instrument as instrument
from pygeodesy.model import Model

from giant.utilities import timefn
from giant.solvers import RidgeRegression

# Define the default options
defaults = {
    'input': None,
    'output': 'sqlite:///detrended.db',
    'cleanonly': False,
    'remove': 'secular, transient',
    'nstd': 3,
    'nvalid': 100,
    't0': 0.0,
    'tf': 3000.0,
    'penalty': 1.0,
    'output_phase': None,
    'output_amp': None,
    'user': 'userCollection.py',
    'iter': 1
}


def detrend(optdict):

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

    # Partition the states
    network.partitionStations()

    # Master reads in the time function collection and initializes an output engine
    if rank == 0:
        collection, iCm = load_collection(network.dates, opts['user'])
        engine_out = Engine(url=opts['output'])
    else:
        collection = iCm = None

    # Broadcast
    collection = comm.bcast(collection, root=0)
    iCm = comm.bcast(iCm, root=0)

    # Create a model for handling the time function
    model = Model(network.dates, collection=collection)

    # Create a solver
    solver = RidgeRegression(model.reg_indices, float(opts['penalty']), regMat=iCm)

    # Determine which components to remove
    parts_to_remove = [s.strip() for s in opts['remove'].split(',')]

    # Loop over components
    n_iter = int(opts['iter'])
    for comp_count, comp in enumerate(inst.components):

        if rank == 0:
            print('%s component' % comp)

        # Read subset data frame
        data_df = network.get(comp, network.sub_names)
        sigma_df = network.get('sigma_' + comp, network.sub_names)

        # Make copies for storing the filtered data
        fit_df = data_df.copy()
        sigma_fit_df = sigma_df.copy()

        # Assign names to data frames to help with writing results
        for df, name in ((data_df, comp), (sigma_df, 'sigma_' + comp),
            (fit_df, 'filt_' + comp), (sigma_fit_df, 'sigma_filt_' + comp)):
            df.name = name

        # Loop over stations
        keep_stations = []
        nstd = int(opts['nstd'])
        seasonal_dat = {}
        for statname in network.sub_names:

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
                    outlierInd = np.abs(dat) > 400.0
                else:
                    outlierInd = np.abs(dat) > 1000.0
                dat[outlierInd] = np.nan

                # Construct subset indices for inversion
                ind = np.isfinite(dat) * np.isfinite(wgt)
                nvalid = ind.nonzero()[0].size
                if nvalid < int(opts['nvalid']):
                    print('Skipping', statname)
                    isStatGood = False
                    break

                # Perform least squares
                mvec = model.invert(solver, dat, wgt=wgt)

                # Model performs reconstruction
                recon = model.predict(mvec)

                # Compute misfit and standard deviation
                misfit = dat - recon['full']
                stdev = np.nanstd(misfit)
                print(' - sigma: %f   %d-sigma: %f' % (stdev, nstd, nstd*stdev))

                # Detrend
                filt_signal = model.detrend(dat, recon, parts_to_remove)

                # Attempt to get phase for annual sinusoid in days
                amp, phs = model.computeSeasonalAmpPhase()
                seasonal_dat[statname] = (amp, phs)

                # Update fits and sigmas
                fit_df[statname] = filt_signal
                sigma_fit_df[statname] = recon['sigma']
                
                # Remove outliers
                outlierInd = np.abs(misfit) > (nstd * stdev)
                dat[outlierInd] = np.nan

            if isStatGood:
                keep_stations.append(statname)

            ## If we are only cleaning the data, then we add back the secular and transient
            #if inputs.cleanFlag and isStatGood:
            #    stat[comp] += stat['secular_' + comp] + stat['transient_' + comp]
            #    stat['secular_' + comp] = 0.0
            #    stat['transient_' + comp] = 0.0

        # Keep only the good stations
        results = []
        for df, name in ((data_df, comp), (sigma_df, 'sigma_' + comp),
            (fit_df, 'filt_' + comp), (sigma_fit_df, 'sigma_filt_' + comp)):
            sub_df = df[keep_stations]
            sub_df['DATE'] = network.dates
            sub_df.name = name
            results.append(sub_df)
        Ndf = len(results)

        # Send my results to the master for writing
        if rank == 0:

            # Get results from each processor and add to table 
            for pid in range(1, size):

                # Get the data
                proc_results = comm.recv(source=pid, tag=77)
                proc_stations = comm.recv(source=pid, tag=87)
                proc_seasonal = comm.recv(source=pid, tag=97)

                # Update list of stations and seasonal data
                keep_stations.extend(proc_stations)
                seasonal_dat.update(proc_seasonal)

                # Update the data frames
                for i in range(Ndf):
                    name = results[i].name
                    results[i] = pd.merge(results[i], proc_results[i],
                        how='outer', on='DATE')
                    results[i].name = name

            # Write to database after getting from all workers
            for i in range(Ndf):
                results[i].to_sql(results[i].name, engine_out.engine, if_exists='replace')

            # Also load the metadata and keep the good stations
            meta = engine.meta()
            meta_sub = meta[np.in1d(meta['id'].values, keep_stations)].reset_index(drop=True)
            meta_sub.to_sql('metadata', engine_out.engine, if_exists='replace')

            # Write seasonal phases if requested
            if opts['output_phase'] is not None and comp == 'up':
                with open(opts['output_phase'], 'w') as pfid:
                    for statname, (amp,phs) in seasonal_dat.items():
                        stat_meta = meta_sub.loc[meta_sub['id'] == statname]
                        pfid.write('%f %f %f 0.5\n' % (float(stat_meta['lon']),
                            float(stat_meta['lat']), phs))

            # Write seasonal amps if requested
            if opts['output_amp'] is not None and comp == 'up':
                with open(opts['output_amp'], 'w') as afid:
                    for statname, (amp,phs) in seasonal_dat.items():
                        stat_meta = meta_sub.loc[meta_sub['id'] == statname]
                        afid.write('%f %f %f 0.5\n' % (float(stat_meta['lon']),
                            float(stat_meta['lat']), amp))
     

        else:
            comm.send(results, dest=0, tag=77)
            comm.send(keep_stations, dest=0, tag=87)
            comm.send(seasonal_dat, dest=0, tag=97)
            del data_df, sigma_df, fit_df, sigma_fit_df

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
        collection = loadDefaultCollection(dates)
    npar = len(collection)

    # Also try to build a prior covariance matrix
    try:
        collfun = imp.load_source('build', inputs.user)
        Cm = collfun.computeCm(collection)
        iCm = np.linalg.inv(Cm)
    except:
        iCm = np.eye(npar)

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
