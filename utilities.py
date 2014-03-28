
import numpy as np
import matplotlib.pyplot as plt


def subsetData(tobs, inputDict, t0=0.0, tf=3000.0, minValid=1, checkOnly=False, ndays=None):
    """
    Subsets GPS data based on a window of observation times.
    """
    # Boolean array of valid observation times
    tbool = tobs >= t0
    if ndays is None:
        tbool *= tobs <= tf
    else:
        beg_ind = tbool.nonzero()[0][0]
        end_ind = beg_ind + ndays
        tbool = np.arange(beg_ind, end_ind, dtype=int)
    print('Subset time window:', tobs[tbool][0], '->', tobs[tbool][-1])

    # Get the station names
    statnames = [key for key in inputDict]

    # Subset data
    if not checkOnly:
        tobs = tobs[tbool]
    nvalidsss = []
    for statname in statnames:
        stat = inputDict[statname]
        # Test to see if station has enough valid data
        indValid = np.isfinite(stat.east[tbool]).nonzero()[0]
        if indValid.size < minValid:
            del inputDict[statname]
        else:
            if checkOnly:
                continue
            else:
                stat.east = stat.east[tbool]
                stat.north = stat.north[tbool]
                stat.up = stat.up[tbool]
                stat.w_n = stat.w_n[tbool]
                stat.w_e = stat.w_e[tbool]
                stat.w_u = stat.w_u[tbool]
                nvalidsss.append(indValid.size)

    #plt.plot(nvalidsss, 'o'); plt.show(); assert False

    return tobs
