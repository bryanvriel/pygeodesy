#-*- coding: utf-8 -*-

import numpy as np
import simplekml
import sys


def make_kml(engine, output):
    """
    Make a kml file of station coordinates for a given engine.
    """

    meta_df = engine.meta()
    lon = meta_df['lon'].values
    lat = meta_df['lat'].values
   
    kml = simplekml.Kml()
    for slon, slat in zip(lon, lat):

        point = kml.newpoint(coords=[(slon, slat)])
        point.style.iconstyle.color = 'r'
        point.style.iconstyle.scale = 1.0
        point.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/pal2/icon26.png'

    kml.savekmz(output) 
