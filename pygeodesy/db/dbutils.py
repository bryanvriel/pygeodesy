#-*- coding: utf-8 -*-

import numpy as np
import datetime
import os

def buildFileList(input_dir):
    """
    Build a file list of data files given an input directory.
    """
    print('Building file list')

    # Initialize empty dictionary for every station
    statdict = {}

    # Traverse the directories
    for root, dirs, files in os.walk(input_dir):
        for file in files:

            if not file.endswith('.tseries'):
                continue

            # Parse the file name for metadata tags
            metlist = file.split('.')
            status = metlist[-3]
            if status != 'final':
                continue
            proctag = int(metlist[-2])
            year = int(metlist[-6])
            month = int(metlist[-5])
            day = int(metlist[-4])
            statname = metlist[0][:4].lower()

            # Convert date to day-of-year
            yearStart = datetime.date(year, 1, 1)
            current = datetime.date(year, month, day)
            doy = (current - yearStart).days + 1

            # Check if station is in dictionary and update its file information
            key = '%4d-%03d' % (year, doy)
            if statname not in statdict:
                filedict = {}
                filedict[key] = (os.path.join(root, file), proctag)
                statdict[statname] = filedict
            else:
                filedict = statdict[statname]
                if key in filedict.keys():
                    fname, oldtag = filedict[key]
                    if proctag > oldtag:
                        filedict[key] = (os.path.join(root, file), proctag)
                else:
                    filedict[key] = (os.path.join(root, file), proctag)

    # Kick out any stations that don't have more than 40 files
    statnames = sorted(list(statdict.keys()))
    removestat = []
    for statname in statnames:
        filedict = statdict[statname]
        if len(filedict) < 40:
            removestat.append(statname)
    print('Skipping these stations:', removestat)

    # Write filenames to text file
    output = 'file_list.txt'
    with open(output, 'w') as fid:
        for statname in statnames:
            if statname in removestat:
                continue
            filedict = statdict[statname]
            keys = sorted(list(filedict.keys()))
            for key in keys:
                filename, proctag = filedict[key]
                fid.write('%s\n' % filename)

    return output


# end of file
