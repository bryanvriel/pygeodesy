#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from mpi4py import MPI
from cvxopt import sparse, matrix, spmatrix
import pickle
import sys
import os

from pygeodesy.db.Engine import Engine
from pygeodesy.model import Model

from giant.utilities import timefn


def partitionData(solver, network, opts, comm):
    """
    Initialize ADMM data.
    """

    rank = comm.Get_rank()
    size = comm.Get_size()

    # --------------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------------

    # Master work
    if rank == 0:
        # Load all data into contiguous arrays
        datArr, wgtArr = network.getDataArrays(order='columns', sigmas=opts['sigmas'],
            components=[opts['component']], scale=float(opts['scale']))
        # Read in time function
        collection, iCm = load_collection(network.dates, opts['user'])
    else:
        datArr = wgtArr = collection = iCm = None

    if rank == 0: print('Broadcasting data')
    datArr = comm.bcast(datArr, root=0)
    wgtArr = comm.bcast(wgtArr, root=0)
    collection = comm.bcast(collection, root=0)
    iCm = comm.bcast(iCm, root=0)

    # Create a model for handling the time function
    model = Model(network.dates, collection=collection)

    # Retrieve problem size parameters
    nstat = network.nstat
    # Number of dictionary parameters
    ndat_time, npar_temporal = model.G.shape
    # Compute global problem size
    NDAT = ndat_time * nstat
    NPAR = npar_temporal * nstat
    if rank == 0: print('Global problem size: (%d x %d)' % (NDAT, NPAR))
    comm.Barrier()

    # Determine partitioning via number of temporal parameters
    sendcnts = getSendcnts(npar_temporal, comm)
    # Get my local temporal problem size
    procN_temporal = sendcnts[rank]
    jstart = sum(sendcnts[:rank])
    jend = jstart + procN_temporal
    G_template = model.G[:,jstart:jend]
    # And local optimization size
    istart = jstart * nstat
    iend = jend * nstat

    # Calculate local "cutoff"
    regF = np.zeros(npar_temporal, dtype=int)
    regF[model.reg_indices] = 1
    local_cutoff = (regF[jstart:jend] < 1).nonzero()[0].size
    # Global global "cutoff"
    regF_sub = np.repeat(regF, nstat)[istart:iend]
    cutoff_ind = (regF_sub < 1).nonzero()[0]
    solver.cutoff = len(cutoff_ind) if len(cutoff_ind) > 0 else 0

    solver.NPAR_GLOBAL = NPAR
    solver.procN = procN_temporal * nstat
    print('Rank %03d problem size: (%d x %d)' % (rank, NDAT, solver.procN))
    print('Rank %03d local: %5d global: %5d' % (rank, local_cutoff, solver.cutoff))

    # --------------------------------------------------------------------------------
    # Make design matrix for subproblem
    # --------------------------------------------------------------------------------

    if rank == 0: print('Building G')
    rows = []; cols = []; Gdat = []; dlist = []; iCd = []
    dat_offset = numGood = 0
    for k in range(ndat_time):

        # Add finite data
        kdat = datArr[k,:]
        finite = np.isfinite(kdat).nonzero()[0]
        dlist.extend(kdat[finite])

        # Add uncertainties
        kvariance = wgtArr[k,finite]**2
        iCd.extend(kvariance)

        # Loop over the station-components to get indices and data
        for cnt, statInd in enumerate(finite):
            rows.extend([cnt + dat_offset] * procN_temporal)
            cols.extend(list(range(statInd, procN_temporal*nstat, nstat)))
            Gdat.extend(G_template[k,:])

        # Update index offset 
        dat_offset += len(finite)

    # Convert lists to numpy arrays
    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    Gdat = np.array(Gdat)

    # Make a sparse matrix
    print(' - making sparse matrices. I and J lengths:', len(rows), len(cols))
    solver.Amat = spmatrix(Gdat, rows, cols, size=(dat_offset,solver.procN))
    solver.b = np.array(dlist)
    solver.iCd = spmatrix(np.array(iCd), range(len(dlist)), range(len(dlist)))
    solver.ndata = len(dlist)
    assert len(dlist) == dat_offset, 'Mismatch: %d vs %d' % (len(dlist), dat_offset)

    del rows, cols, Gdat, dlist, iCd

    # --------------------------------------------------------------------------------
    # Make smoothing matrix for my subproblem
    # --------------------------------------------------------------------------------

    if rank == 0: print('Making smoothing matrix')

    # Compute station distances and weighting matrix for all stations
    if opts['correlation_length'] is not None:
        L0 = float(opts['correlation_length'])
    else:
        L0 = None
    weightMat = network.computeNetworkWeighting(smooth=float(opts['oversmooth']), L0=L0)
    
    # Initialize array for local prior covariance matrix
    D = np.zeros((solver.procN,solver.procN))

    # Determine indices for each parameter and fill with values
    istart = 0
    for j in range(procN_temporal):
        iend = istart + nstat
        D[istart:iend,istart:iend] = weightMat
        istart += nstat

    # Finally, convert to CVXOPT sparse matrix and save to solver
    solver.D = sparse(matrix(D.T.tolist()))
    
    return model


def distributeSolutions(mvec, engine, model, network, opts):
    """
    Distribute ADMM solutions to the correct stations and save to database.
    """

    # Cache some options
    scale = float(opts['scale'])
    component = opts['component']

    # First write original data to database
    df = network.get(component, None, scale=scale, with_date=True)
    df.to_sql(component, engine.engine, if_exists='replace')

    # Initialize summary dictionary
    results = {'secular': {'DATE': network.dates}, 
        'seasonal': {'DATE': network.dates}, 
        'step': {'DATE': network.dates},
        'transient': {'DATE': network.dates}, 
        'full': {'DATE': network.dates}}

    # Dictionary for coefficients
    coeffs = {}
    
    # Loop over stations
    nstat = network.nstat
    for cnt, statname in enumerate(network.names):
        # Make predictions
        mstat = mvec[cnt::nstat]
        fit_dict = model.predict(mstat)
        for ftype in ('secular', 'seasonal', 'step', 'transient', 'full'):
            results[ftype][statname] = fit_dict[ftype]
        # Save coefficients
        coeffs[statname] = mstat

    # Write results to database
    for ftype in ('secular', 'seasonal', 'step', 'transient', 'full'):
        df = pd.DataFrame(results[ftype])
        df.to_sql('%s_%s' % (ftype, component), engine.engine, 
            if_exists='replace')

    # Also for coefficients
    coeffs = pd.DataFrame(coeffs)
    coeffs.to_sql('coeff_%s' % component, engine.engine, if_exists='replace')

    return


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


def getSendcnts(N, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    nominal_load = N // size
    if rank == size - 1:
        procN = N - rank * nominal_load
    else:
        procN = nominal_load
    sendcnts = comm.allgather(procN)
    return sendcnts


def saveData(solver, model, rank, component):
    """
    Save data.
    """
    if not os.path.isdir('PICKLE'):
        os.mkdir('PICKLE')
    proto = pickle.HIGHEST_PROTOCOL
    with open('PICKLE/data_%s_proc%03d.pkl' % (component, rank), 'wb') as fid:
        pickle.dump(solver, fid, protocol=proto)
        pickle.dump(model, fid, protocol=proto)
    return


def loadData(rank, component, comm):
    """
    Load data.
    """
    print('Rank %03d loading data' % rank)
    with open('PICKLE/data_%s_proc%03d.pkl' % (component, rank), 'rb') as fid:
        solver = pickle.load(fid)
        model = pickle.load(fid)
    # Give solver a fresh communicator
    solver.comm = comm
    return solver, model


def updatePenalties(solver, opts, model, rank, component):
    """
    Check if penalties have been changed. If so, re-initialize CVXOPT data.
    """
    sparse = float(opts['sparse_penalty'])
    smooth = float(opts['smooth_penalty'])
    if (abs(solver.sparsePenalty - sparse) > 1.0e-5 or
        abs(solver.smoothPenalty - smooth) > 1.0e-5):
        if rank == 0: print('Updating CVXOPT data')
        solver.prepareCVXOPT(sparse, smooth, DtD=False)
        saveData(solver, model, rank, component)

    return


def ALS_factor(A, beta, num_features=None, max_step=30):

    # Determine problem size
    nobs, nstat = A.shape
    n_feat = num_features or nstat

    # Initial values are random numbers
    spatialMat = np.random.standard_normal((nstat,n_feat))
    temporalMat = np.random.standard_normal((nobs,n_feat))
    betaReg = beta * np.eye(n_feat)

    # Train
    step = 0
    errors = []
    while step < max_step:

        # Construct problem matrix for spatial components
        lhs = np.dot(temporalMat.T, temporalMat)
        rhs = np.zeros((n_feat,nstat))
        for j in range(nstat):
            data_col = A[:,j]
            ind = np.isfinite(data_col)
            rhs[:,j] = np.dot(temporalMat[ind,:].T, data_col[ind])

        # Solve
        spatialMat = 2.0 * np.linalg.lstsq(betaReg + 2.0*lhs, rhs)[0]
        spatialMat = spatialMat.T

        # Construct problem matrix for temporal components
        lhs = np.dot(spatialMat.T, spatialMat)
        rhs = np.zeros((n_feat,nobs))
        for i in range(nobs):
            data_row = A[i,:]
            ind = np.isfinite(data_row)
            rhs[:,i] = np.dot(spatialMat[ind,:].T, data_row[ind])

        # Solve
        temporalMat = 2.0 * np.linalg.lstsq(betaReg + 2.0*lhs, rhs)[0]
        temporalMat = temporalMat.T

        # Compute prediction error
        prediction = np.dot(temporalMat, spatialMat.T)
        misfit = (A - prediction).flatten()
        ind = np.isfinite(misfit)
        error = np.dot(misfit[ind], misfit[ind])
        print(' - ALS iteration %02d error: %f' % (step, error))

        # Update
        errors.append(error)
        step += 1

    return temporalMat, spatialMat, errors
   

# end of file 
