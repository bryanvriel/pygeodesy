#-*- coding: utf-8 -*-

import numpy as np
import tsinsar as ts
import os

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
        self.success = True
        if 'sopac' in format:
            fname = '%s/%s.neu' % (gpsdir, stname + fileKernel)
            if os.path.isfile(fname):
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
            else:
                self.success = False

        elif 'pbo' in format:
            fname = '%s/%s.pbo.final_igs08.pos' % (gpsdir, stname.upper())
            if os.path.isfile(fname):
                fid = open(fname, 'r')
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
            else:
                self.success = False

        elif 'geonetnz' in format:
            fname = '%s/%s_neu.dat' % (gpsdir, stname.upper())
            if os.path.isfile(fname):
                fid = open(fname, 'r')
                self.tdec, north, east, up = np.loadtxt(fid, unpack=True)
                self.north, self.east, self.up = [dataFactor * val for val in [north, east, up]]
                ones = np.ones_like(self.north, dtype=float)
                self.sn, self.se, self.su = ones, ones, ones
                fid.close()
            else:
                self.success = False

        elif 'usgs' in format:
            fname = '%s/%s.rneu' % (gpsdir, stname.lower())
            if os.path.isfile(fname):
                fid = open(fname, 'r')
                data = []
                for line in fid:
                    linedat = line.split()
                    t = float(linedat[1])
                    e,n,u = [float(val) for val in linedat[2:5]]
                    se,sn,su = [float(val)**2 for val in linedat[6:9]]
                    data.append([t,e,n,u,se,sn,su])
                fid.close()
                data = np.array(data)
                self.tdec = data[:,0]
                self.north, self.east, self.up = data[:,1], data[:,2], data[:,3]
                self.sn, self.se, self.su = data[:,4], data[:,5], data[:,6]
            else:
                self.success = False

        elif 'gam' in format:
            fname = '%s/%s.gam' % (gpsdir, stname)
            if os.path.isfile(fname):
                t,e,n,se,sn,u,su = np.loadtxt(fname, usecols=(0,1,2,3,4,6,7), unpack=True)
                self.tdec = t
                self.north, self.east, self.up = n, e, u
                self.sn, self.se, self.su = sn, se, su
            else:
                self.success = False
                        
        else:
            assert txtreader is not None, 'No reader specified for GPS data'
            tdec,north,east,up,sn,se,su = txtreader(stname, gpsdir)
            self.tdec = tdec
            self.north, self.east, self.up = north, east, up
            self.sn, self.se, self.su = sn, se, su 

        return

