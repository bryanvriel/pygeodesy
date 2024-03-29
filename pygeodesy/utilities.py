#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path


def merge(objtup, verbose=False):
    """
    Merges multiple TimeSeries objects.
    """
    import copy

    # Reference first object and use that to extend
    objout = objtup[0]
    ref_type = objout.dtype
    ref_tdec = objout.tdec

    # Loop over the rest of the objects
    if verbose: print('')
    for obj in objtup[1:]:

        if verbose: print('Interpolating', obj.name)

        # Make sure it's the same type
        assert obj.dtype == ref_type

        # Add dictionary items after interpolation
        for statname, rawstat in obj.statGen:
            stat = copy.deepcopy(rawstat)
            for comp in obj.components:
                dat = stat[comp]
                wgt = stat['w_' + comp]
                stat[comp] = np.interp(ref_tdec, obj.tdec, dat, left=np.nan,
                    right=np.nan)
                stat['w_' + comp] = np.interp(ref_tdec, obj.tdec, wgt, left=np.nan,
                    right=np.nan)
                try:
                    stat['filt_' + comp] = np.interp(ref_tdec, obj.tdec, stat['filt_' + comp],
                        left=np.nan, right=np.nan)
                except KeyError:
                    pass
            objout.statDict[statname] = stat

    # Remake the station generator
    objout.makeStatGen()
    objout.transferDictInfo()

    return


def selectDataClass(h5file):
    """
    Utility function to read in a stack file, get its data type, and return
    an instance of the appropriate class. If no data dtype is found in the 
    stack file, we assume GPS data.
    """
    import h5py
    with h5py.File(h5file, 'r') as fid:
        try:
            stype = fid['dtype'].value
            if stype == 'wells':
                from .Wells import Wells
                data = Wells()
            elif stype == 'gps':
                from .GPS import GPS
                data = GPS()
            elif stype == 'insar':
                from giant import Insar
                data = Insar()
        except KeyError:
            from .GPS import GPS
            data = GPS()

    return data


def subsetDataWithPoly(inputDict, points, h5=True):
    """
    Subset GPS stations within a polynomial.
    """

    # Make a path out of the polynomial points
    poly = Path(points)

    # Figure out which stations lie inside the polygon
    keepstat = ['tdec', 'G', 'cutoff', 'npbspline']
    for statname, stat in inputDict.items():
        if statname in ['tdec', 'G', 'cutoff']:
            continue
        if h5:
            test = poly.contains_points(np.array([[stat['lon'], stat['lat']]]))[0]
        else:
            test = poly.contains_points(np.array([[stat.lon, stat.lat]]))[0]
        if test:
            keepstat.append(statname)

    # Remove the ones that don't
    statnames = list(inputDict.keys())
    for statname in statnames:
        if statname.lower() not in keepstat:
            del inputDict[statname]

    return


def subsetData(tobs, inputDict, t0=0.0, tf=3000.0, minValid=1, checkOnly=False, ndays=None,
               statlist=None, subfactor=1, h5=True):
    """
    Subsets GPS data based on a window of observation times.
    """
    # First check if a list of stations to keep is provided
    statnames = [name for name in inputDict.keys() if name not in 
        ['tdec', 'G', 'cutoff', 'npbspline', 'dtype']]
    if statlist is not None:
        for statname in statnames:
            if statname.lower() not in statlist:
                del inputDict[statname]
    statnames = [name for name in inputDict.keys() if name not in 
        ['tdec', 'G', 'cutoff', 'npbspline', 'dtype']]

    # Boolean array of valid observation times
    tbool = tobs >= t0
    if ndays is None:
        tbool *= tobs <= tf
        tbool = tbool.nonzero()[0][::subfactor]
    else:
        beg_ind = tbool.nonzero()[0][0]
        end_ind = beg_ind + ndays
        tbool = np.arange(beg_ind, end_ind, dtype=int)[::subfactor]
    print('Subset time window:', tobs[tbool][0], '->', tobs[tbool][-1])

    # Subset data
    if not checkOnly:
        tobs = tobs[tbool]
    nvalidsss = []
    for statname in statnames:
        stat = inputDict[statname]
        if statname == 'tdec': continue
        # Test to see if station has enough valid data
        if h5:
            dat = np.array(stat['up'])
        else:
            dat = stat.east
        indValid = np.isfinite(dat[tbool]).nonzero()[0]
        if indValid.size < minValid:
            del inputDict[statname]
        else:
            if checkOnly:
                continue
            else:
                # First get list of station attributes
                if h5:
                    attrlist = list(stat.keys())
                else:
                    attrlist = dir(stat)
                for attr in ('east', 'north', 'up', 'w_east', 'w_north', 'w_up', 'status'):
                    if attr not in attrlist:
                        continue
                    if h5:
                        dat = np.array(stat[attr])
                        stat[attr] = dat[tbool]
                        for modifier in ['filt_', 'secular_', 'transient_']:
                            try:
                                filtdat = np.array(stat[modifier + attr])
                                stat[modifier + attr] = filtdat[tbool]
                            except KeyError:
                                pass
                            except IndexError:
                                pass
                    else:
                        dat = getattr(stat, attr)
                        setattr(stat, attr, dat[tbool])
                        try:
                            filtdat = getattr(stat, 'filt_' + attr)
                            setattr(stat, 'filt_' + attr, filtdat[tbool])
                        except AttributeError:
                            pass
                nvalidsss.append(indValid.size)

    return tobs


def partitionStations(data, comm=None, strategy='stations'):
    """
    Utility function for determining partitioning strategy.
    """
    # Save the communicator and get rank and size
    from mpi4py import MPI
    comm = comm or MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Determine partitioning strategy
    if isinstance(strategy, int):
        N = strategy
    elif isinstance(strategy, str):
        if strategy == 'stations':
            N = data.nstat
        elif strategy == 'total':
            N = data.nstat * data.ncomp
        else:
            raise NotImplementedError('Unsupported partitioning strategy')
    else:
        raise NotImplementedError('Argument strategy must be int or str')

    # Do the partitioning
    nominal_load = N // size
    if rank == size - 1:
        procN = N - rank * nominal_load
    else:
        procN = nominal_load
    sendcnts = comm.allgather(procN)

    return sendcnts


def appendSeasonalDictionary(G, G_mod_ref, data):
    """
    Prepends seasonal temporal dictionary to an existing temporal dictionary. The 
    seasonal dictionary is represented by modulating integrated B-splines for a template
    of repeating B-splines.
    """
    import tsinsar as ts
    npbspline = data.npbspline
    G_seas = ts.Timefn([['PBSPLINES',[3],[npbspline],1.0]], data.trel)[0]

    template = np.dot(G_seas, self.m_seas[jj,:])
    # Normalize the template
    mean_spline = np.mean(template)
    fit_norm = template - mean_spline
    template = fit_norm / (0.5*(fit_norm.max() - fit_norm.min()))
    G_mod = G_mod_ref.copy()
    for nn in range(nsplmod):
        G_mod[:,nn] *= template
    G = np.column_stack((G_mod, self.G))


# Some fast matrix multiply routines from GIAnT
def dmultl(dvec, mat):
    """
    Left multiply with a diagonal matrix. Faster.
    
    .. Args:
        
        * dvec    -> Diagonal matrix represented as a vector
        * mat     -> Matrix
        
    .. Returns:
    
        * res    -> dot (diag(dvec), mat)
    """

    res = (dvec*mat.T).T
    return res

def dmultr(mat, dvec):
    """
    Right multiply with a diagonal matrix. Faster.
    
    .. Args:
        
        * dvec    -> Diagonal matrix represented as a vector
        * mat     -> Matrix
        
    .. Returns:
    
        * res     -> dot(mat, diag(dvec))
    """

    res = dvec*mat
    return res   


def datestr2tdec(yy=0, mm=0, dd=0, hour=0, minute=0, sec=0, microsec=0,
                 datestr=None, pydtime=None):
    """
    Convert year, month, day, hours, minutes, seconds to decimal year.
    """
    import datetime
    if datestr is not None:
        yy, mm, dd = [int(val) for val in datestr.split('-')]
        hour, minute, sec = [0, 0, 0]

    if pydtime is not None:
        attrs = ['year', 'month', 'day', 'hour', 'minute', 'second']
        yy, mm, dd, hour, minute, sec = [getattr(pydtime, attr) for attr in attrs]


    # Make datetime object for start of year
    yearStart = datetime.datetime(yy, 1, 1, 0, 0, 0)
    # Make datetime object for input time
    current = datetime.datetime(yy, mm, dd, hour, minute, sec, microsec)
    # Compute number of days elapsed since start of year
    tdelta = current - yearStart
    # Convert to decimal year and account for leap year
    if yy % 4 == 0:
        return float(yy) + tdelta.total_seconds() / (366.0 * 86400)
    else:
        return float(yy) + tdelta.total_seconds() / (365.0 * 86400)


# end of file 
