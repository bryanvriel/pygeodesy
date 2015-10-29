
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path


def subsetDataWithPoly(inputDict, points, h5=True):
    """
    Subset GPS stations within a polynomial.
    """

    # Make a path out of the polynomial points
    poly = Path(points)

    # Figure out which stations lie inside the polygon
    keepstat = ['tdec', 'G', 'cutoff']
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
    statnames = list(inputDict.keys())
    if statlist is not None:
        for statname in statnames:
            if statname.lower() not in statlist:
                del inputDict[statname]
    statnames = list(inputDict.keys())

    # Boolean array of valid observation times
    tbool = tobs >= t0
    if ndays is None:
        tbool *= tobs <= tf
        tbool = tbool.nonzero()[0][::subfactor]
    else:
        beg_ind = tbool.nonzero()[0][0]
        end_ind = beg_ind + ndays
        tbool = np.arange(beg_ind, end_ind, dtype=int)[::subfactor]
    print(('Subset time window:', tobs[tbool][0], '->', tobs[tbool][-1]))

    # Subset data
    if not checkOnly:
        tobs = tobs[tbool]
    nvalidsss = []
    for statname in statnames:
        stat = inputDict[statname]
        if statname == 'tdec': continue
        # Test to see if station has enough valid data
        if h5:
            dat = np.array(stat['east'])
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
                        try:
                            filtdat = np.array(stat['filt_' + attr])
                            stat['filt_' + attr] = filtdat[tbool]
                        except KeyError:
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
