#-*- coding: utf-8 -*-

from .GPS import GPS
from .Wells import Wells
from .EDM import EDM


def select(instrument_type, fmt=None):
    """
    Return the correction instrument object for a specified type.
    """

    # Initialize the correct instrument
    if instrument_type in ['gps', 'GPS', 'Gps']:
        obj = GPS()
    elif instrument_type in ['edm', 'EDM', 'Edm']:
        obj = EDM()
    elif instrument_type in ['wells', 'Wells']:
        obj = Wells()
    else:
        raise NotImplementedError('Unsupported instrument type.')

    # Set the format
    obj.datformat = fmt
    return obj


# end of file
