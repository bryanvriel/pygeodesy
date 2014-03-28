
import numpy as np

class frep:
    '''Functional representation of the SOPAC model parameters.'''
    def __init__(self):
        '''Setting up any model string.'''
        self.amp = None
        self.err = None
        self.rep = None
        self.win = None

    def setslope(self,slope,err,t0,t1):
        self.amp = np.float(slope)
        self.err = np.float(err)
        self.win = [np.float(t0),np.float(t1)]
        self.rep = ['LINEAR',[np.float(t0)]]

    def setdecay(self,amp,err,t0,tau):
        self.amp = -1.0 * np.float(amp)
        self.err = np.float(err)
        self.win = [np.float(t0),np.inf]
        self.rep = ['EXP',[np.float(t0)],[np.float(tau)/365.25]] 

    def setoffset(self,amp,err,t0):
        self.amp = np.float(amp)
        self.err = np.float(err)
        self.win=[np.float(t0),np.inf]
        self.rep = ['STEP', [np.float(t0)]]

    def setannual(self,amp,err,phase):
        ph = np.float(phase)
        gph = np.array([np.cos(ph),np.sin(ph)])
        self.amp = np.float(amp)*gph
        self.err = np.float(err)*np.abs(gph)
        self.win =[-np.inf, np.inf]
        self.rep = ['SEASONAL',[1.0]]
        
    def setsemi(self,amp,err,phase):
        ph = np.float(phase)
        gph = np.array([np.cos(ph),np.sin(ph)])
        self.amp = np.float(amp)*gph
        self.err = np.float(err)*np.abs(gph)
        self.win =[-np.inf, np.inf]
        self.rep = ['SEASONAL',[0.5]]

    def __str__(self):
        return str([self.win, self.amp, self.err, self.rep])


class model:
    '''Putting together a model of the GPS station.'''
    def __init__(self, dlines):
        '''Initiates a model with input lines from header of each station file.'''

        self.slope = []
        self.decay = []
        self.offset = []
        self.annual = []
        self.semi = []

        for line in dlines:

            if '-------------------------------------------' in line:
                break

            emptyline = False
            if ':' in line:
                allPrts = line.split(':')
                if len(allPrts) == 2:
                    label, parts = allPrts
                elif len(allPrts) == 3:
                    label = allPrts[0]
                    parts = allPrts[1] + allPrts[2]
            else:
                continue

            if 'Scaling factor' in label:
                continue
       
            val, parts_str = parts.split('+/-')
            val = float(val)
            parts = parts_str.split()

            if 'slope' in label:
                npart = frep()
                npart.setslope(val, parts[0], parts[3], parts[6])
                self.slope.append(npart)

            elif 'ps' in label:
                continue
                if parts[2] == 'postseismic':
                    continue
                npart = frep()
                npart.setdecay(parts[3],parts[5],parts[7],parts[9])
                self.decay.append(npart)

            elif 'offset' in label:
                npart = frep()
                npart.setoffset(val, parts[0], parts[3])
                self.offset.append(npart)

            elif 'annual' in label:
                npart = frep()
                if 'semi-annual' in label:
                    npart.setsemi(val, parts[0], parts[3])
                    self.semi.append(npart)
                else:
                    npart.setannual(val, parts[0], parts[3])
                    self.annual.append(npart)

            else:
                emptyline = True

        return

             

class sopac:
    '''Class to deal with model description provided in the SOPAC daily solutions. '''
    def __init__(self, fname):
        '''Reads and initiates the model parameters.'''
        dlines = self.read_header(fname)
        nind = np.where([k=='n component' for k in dlines])
        nind = nind[0][0]

        eind = np.where([k=='e component' for k in dlines])
        eind = eind[0][0]

        uind = np.where([k=='u component' for k in dlines])
        uind = uind[0][0]

        self.north = model(dlines[nind:eind])
        self.east = model(dlines[eind:uind])
        self.up = model(dlines[uind:])
        

    @staticmethod
    def read_header(fname):
        '''Reads in all lines in the text file starting with specified character.

         .. Args:

            * fname           Text file name to read.

        .. Returns:

            * dlines         Lines with specified starting pattern.'''

        # Open the file
        fid = open(fname,'r')
        lines = []

        ch = '#'
        nch = len(ch)
        for line in fid:
            line = line.rstrip()
            if line[0:nch] == ch:
                line = line[nch:].lstrip()
                line= line.translate(None,'#;*()[]!')
                line = line.lstrip()
                line = line.rstrip()
                if line != '':
                    lines.append(line)

        return lines


############################################################
# Program is part of GIAnT v1.0                            #
# Copyright 2012, by the California Institute of Technology#
# Contact: earthdef@gps.caltech.edu                        #
############################################################
