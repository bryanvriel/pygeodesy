#-*- coding: utf-8

import pygeodesy as pg
import pandas as pd
import datetime
import pyre
import sys
import os

class MakeDB(pg.components.task, family='pygeodesy.makedb'):
    """
    Make a time series database.
    """

    columns = pyre.properties.int(default=None)
    columns.doc = 'Number of columns in ASCII file of raw data'

    directory = pyre.properties.str(default=None)
    directory.doc = 'Directory of raw data'

    dbname = pyre.properties.str(default='data.db')
    dbname.doc = 'Output database filename'

    filelist = pyre.properties.str(default=None)
    filelist.doc = 'Filename of ASCII file of list of data files to process'

    metafile = pyre.properties.str(default=None)
    metafile.doc = 'Filename containing coordinates of station locations'

    metafmt = pyre.properties.str(default=None)
    metafmt.doc = 'Column format of filename containing station coordinates'

    chunk_size = pyre.properties.int(default=100000)
    chunk_size.doc = 'Chunk size of database'

    extension = pyre.properties.str(default='.neu')
    extension.doc = 'File extension of raw data files'

    preprocess = pyre.properties.bool(default=False)
    preprocess.doc = 'Flag for pre-processing raw data (cleaning) before database creation'

    @pyre.export
    def main(self, plexus, argv):
        """
        Main entrypoint to this application.
        """    

        # Read list of files or build
        if self.directory is None and self.filelist is not None:
            filelist = np.loadtxt(self.filelist, dtype=bytes).astype(str)
        elif self.directory is not None and self.filelist is None:
            fname = pb.db.utils.buildFileList(self.directory, self.format, self.extension)
            filelist = np.loadtxt(fname, dtype=bytes).astype(str)
        else:
            assert False, 'Must provide input directory or file list'

        # Initialize engine for SQL database
        engine = pg.db.Engine(dbname=self.dbname, dbtype=self.dbtype)

        # Initialize an instrument
        inst = pg.instrument.select(self.type, fmt=self.format)
        # Set its format for reading ASCII data
        inst.updateASCIIformat(self.format, columns=self.columns)
        # Parse any fixed width strings
        inst.parseFixedWidths(self.fixed_widths)

        # Read a metadata file if provided
        inst.read_metadata_ascii(self.metafile, fmtdict=self.metafmt)
        meta_dict = inst.reformat_metadata(fmt='dict')

        # Initialize the database
        engine.initdb(new=False)

        # Get unique file list
        filelist = engine.getUniqueFiles(newlist=filelist)
        print('Number of new files to add to database:', len(filelist))

        # Make an interface object to link the instrument and SQL table
        interface = pg.db.Interface(inst, engine)
        interface.data_to_table(filelist, meta_dict, chunk_size=self.chunk_size)

        # Update metadata
        interface.update_meta(meta_dict)


# end of file
