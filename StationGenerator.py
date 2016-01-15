#-*- coding: utf-8 -*-


class StationGenerator:
    """
    Custom iterable container class for looping over data for a given dataset.
    """

    def __init__(self, los=None, w_los=None, lat=None, lon=None, elev=None):
        """
        Init of StationGenerator. Saves InSAR data.

        Parameters
        ----------
        los: ndarray
            3D array of shape (Nifg,Ny,Nx) corresponding to InSAR data.
        w_los: ndarray
            3D array of shape (Nifg,Ny,Nx) corresponding to inverse of InSAR 
            uncertainties.
        """
        self.los = los
        self.w_los = w_los
        self.lat = lat
        self.lon = lon
        self.elev = elev
        return


    def __getitem__(self, key):
        """
        Parses key for pixel coordinates.

        Parameters
        ----------
        key: str
            str formatted as '%04d-%04d' corresponding to 'i-j'.

        Return
        ------
        d: dict
            dict containing 'los', 'w_los', 'lon', 'lat', and 'elev' keys.
        """
        # Get pixel coordinates
        i,j = [int(val) for val in key.split('-')]
        # Return full dictionary
        outdict = {}
        for attr in ('los', 'w_los', 'lon', 'lat', 'elev'):
            outdict[attr] = getattr(self, attr)[...,i,j]
        return outdict


    def __setitem__(self, key, value_dict):
        """
        Updates arrays with anything in value dict.

        Parameters
        ----------
        key: str
            str formatted as '%04d-%04d' corresponding to 'i-j'.
        value: dict
            dict containing attributes w/ values to update.
        """
        # Get pixel coordinates
        i,j = [int(val) for val in key.split('-')]
        # Loop over dict items and set attribute
        for attr, value in value_dict.items():
            arr = getattr(self, attr)
            arr[...,i,j] = value
        return
    

    def __iter__(self):
        """
        Defines how to iterate over the pixels in a {TimeSeries}-compatible way.
        Yields a key, dict pair.
        """
        for (i,j) in self.pixel_coordinates:
            # Contstruct output dictionary
            outdict = {}
            for attr in ('los', 'w_los', 'lon', 'lat', 'elev'):
                outdict[attr] = getattr(self, attr)[...,i,j]
            # Yield it
            yield ('%04d-%04d' % (i,j), outdict)


# end of file
