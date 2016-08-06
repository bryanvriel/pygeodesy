#-*- coding: utf-8 -*-

import numpy as np

def xyz2llh(X, deg=False):
    """
    Convert XYZ coordinates to latitude, longitude, height
    """
    # Ellipsoid parameters
    A = 6378137.0
    E2 = 0.0066943799901
    B = np.sqrt(A*A*(1.0 - E2))

    if type(X) is list:
        X = np.array(X)

    x = X[0,...]
    y = X[1,...]
    z = X[2,...]

    # Longitude
    lon = np.arctan2(y, x)

    # Latitude
    p = np.sqrt(x*x + y*y) + 0.1
    alpha = np.arctan(z*A / (p*B))
    numer = z + E2/(1.0 - E2) * B * np.sin(alpha)**3
    denom = p - E2 * A * np.cos(alpha)**3
    lat = np.arctan(numer / denom)

    # Height
    h = p/np.cos(lat) - A/np.sqrt(1.0 - E2*np.sin(lat)**2)

    if deg:
        return lat*180/np.pi, lon*180/np.pi, h
    else:
        return lat, lon, h

# end of file
