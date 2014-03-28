#-*- coding: utf-8 -*-

import numpy as np
import tsinsar as ts

class STN:
    """
    Class to hold information from a single GPS station
    """

    def __init__(self, stname, gpsdir, format='sopac', txtreader=None,
                 fileKernel='CleanFlt', dataFactor=1000.0):
        """
        Initialization of single GPS station. Read in data according to a specified format. If
        'format' is not 'sopac' or 'pbo', must provide a txtreader function that stores time in 
        decimal year and reads in 3-component data and errors in millimeters.
        """

        if 'sopac' in format:
            fname = '%s/%s.neu' % (gpsdir, stname + fileKernel)
            [ddec,yr,day,north,east,up,dnor,deas,dup] = ts.tsio.textread(fname,
                    'F I I F F F F F F')
            self.fname = fname
            self.tdec = ddec
            self.north = north * dataFactor
            self.east = east * dataFactor
            self.up = up * dataFactor
            self.sn = (dnor * dataFactor)**2
            self.se = (deas * dataFactor)**2
            self.su = (dup * dataFactor)**2

        elif 'pbo' in format:
            fname = '%s/%s.pbo.final_igs08.pos' % (gpsdir, stname.upper())
            try:
                fid = open(fname, 'r')
            except IOError:
                print 'skipping', fname
                return None
            line = fid.readline()
            while True:
                line = fid.readline()
                if 'End Field Description' in line:
                    dumm = fid.readline()
                    break
            [mjd,north,east,up,dnor,deas,dup] = np.loadtxt(fid, unpack=True,
                                                           usecols=(2,15,16,17,18,19,20))
            # Crude conversion to decimal year
            mjd -= 53005.5
            self.tdec = mjd / 365.25 + 2004.0
            self.north, self.east, self.up = [dataFactor * val for val in [north, east, up]]
            self.sn, self.se, self.su = [(val * dataFactor)**2 for val in [dnor, deas, dup]]

        elif 'geonetnz' in format:
            fname = '%s/%s_neu.dat' % (gpsdir, stname.upper())
            try:
                fid = open(fname, 'r')
            except IOError:
                print 'skipping', fname
                return None
            self.tdec, north, east, up = np.loadtxt(fid, unpack=True)
            self.north, self.east, self.up = [dataFactor * val for val in [north, east, up]]
            ones = np.ones_like(self.north, dtype=float)
            self.sn, self.se, self.su = ones, ones, ones
            fid.close()
                        
        else:
            assert txtreader is not None, 'No reader specified for GPS data'
   
            tdec,north,east,up,sn,se,su = txtreader(stname, gpsdir)
            self.tdec = tdec
            self.north, self.east, self.up = north, east, up
            self.sn, self.se, self.su = sn, se, su 

        return

