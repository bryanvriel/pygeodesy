#-*- coding: utf-8 -*-

import numpy as np
import datetime as dtime
import pandas as pd
from tqdm import tqdm
import sys

class Interface:
    """
    Class for interfacing a SQL engine with an Instrument in order to read and write
    data to a table.
    """

    def __init__(self, inst, engine):
        """
        Initialize with an instrument and engine.
        """
        self.inst = inst
        self.engine = engine
        return


    def data_to_table(self, filelist, meta_dict, chunk_size=100000):
        """
        Read data from a list of files. The instrument will know the data format, and
        the engine will write the data to a table. Data are stored in memory as a
        data frame until chunk_size observations are read. The engine will write
        the data and clear the data frame.
        """
        # Initialize empty dictionary to store data
        data = self.empty_dict()
    
        # Cache the column indices
        cols = self.inst.columns

        # Use instrument to check if we should parse meta data from headers
        read_header = self.inst.read_header

        # Loop over the files
        obs_cnt = 0
        print('')
        for filecnt, filepath in enumerate(filelist):

            if filecnt % 50 == 0:
                sys.stdout.write(' - file %4d / %4d\r' % (filecnt, len(filelist)))
                sys.stdout.flush()
            
            # Load all the data using numpy into an array of strings
            try:
                all_data = np.atleast_2d(np.loadtxt(filepath, dtype=bytes).astype(str))
            except ValueError:
                print('Wrong number of columns for file', filepath)
                continue
            Nobs = all_data.shape[0]

            # A little string parsing to get the station id
            statname = self.inst.parse_id(filepath)
            data['id'].extend(Nobs * [statname])

            # Store the observation data to the dictionary
            for comp in self.inst.components:
                data[comp].extend(all_data[:,cols[comp]].astype(float))
                data['sigma_'+comp].extend(all_data[:,cols['sigma_'+comp]].astype(float))

            # Get year and hours
            years = all_data[:,cols['year']].astype(int)
            if cols['hour'] is not None:
                hours = all_data[:,cols['hour']].astype(int)
            else:
                hours = Nobs * [0]

            # Make dates from day-of-year or month-day
            if cols['month'] is not None:
                months = all_data[:,cols['month']].astype(int)
                days = all_data[:,cols['day']].astype(int)
                for i in range(Nobs):
                    data['DATE'].append(dtime.datetime(years[i], months[i], days[i], hours[i]))

            elif cols['doy'] is not None:
                doy = all_data[:,cols['doy']]
                for i in range(Nobs):
                    date = dtime.datetime(years[i], 1, 1)
                    data['DATE'].append(date + dtime.timedelta(int(doy[i])-1))

            # Save the file path
            self.engine.addFile(filepath)

            # Read meta data from header if we need to
            if read_header:
                try:
                    self.inst.read_meta_header(filepath, meta_dict=meta_dict)
                except ValueError:
                    print('Trouble reading header for', filepath)
                    pass

            # Write to table if we meet the chunksize
            if obs_cnt > chunk_size:
                self._write_data_table(data)
                data = self.empty_dict()
                obs_cnt = 0

            obs_cnt += Nobs

        print('')

        # Write anything left over and return
        self._write_data_table(data)
        return
           

    def _write_data_table(self, data):
        """
        Write data dictionary to SQL table using a data frame.
        """
        df = pd.DataFrame(data)
        df.to_sql('tseries', self.engine.engine, if_exists='append') 
        return


    def empty_dict(self):
        """
        Return a dictionary of empty data.
        """
        data = {'DATE': [], 'id': []}
        for comp in self.inst.components:
            data[comp] = []
            data['sigma_' + comp] = []
        return data


    def update_meta(self, meta_dict):
        """
        Update metadata table using the engine.
        """
        meta_df = pd.DataFrame(meta_dict)
        ref_meta = self.engine.meta()
        meta_df = pd.concat([meta_df, ref_meta]).drop_duplicates(subset='id')
        meta_df.to_sql('metadata', self.engine.engine, if_exists='replace')
        return


    def subset_table(self, idlist, engine_out, tstart=None, tend=None, filelist=[],
                     scale=1.0, block_size=10):
        """
        Subset a raw data table using a list of station IDs, and perform
        an outer join using the dates.
        """
        from functools import reduce

        print('\nSubsetting network to %d stations' % len(idlist))
        # First trim the file list (if provided) to keep only the files associated 
        # with the list of stations
        print(' - subsetting files first')
        engine_out.addFile(filelist)

        # Loop over the components
        query = "SELECT DATE, %s, sigma_%s FROM tseries WHERE id = '%s';"
        for component in self.inst.components:

            print(' - subsetting component', component)

            # Loop over the stations
            print(' - merging stations')
            data_df = []; sigma_df = []
            for i, statname in enumerate(idlist):
                
                # Query database for current station+component 
                data = pd.read_sql_query(query % (component, component, statname),
                                         self.engine.engine)
                new_data_df = data[['DATE', component]]
                new_sigma_df = data[['DATE', 'sigma_' + component]]

                # Use station name in columns as a unique identifier
                new_data_df.columns = ['DATE', statname]
                new_sigma_df.columns = ['DATE', statname]

                # Scale the data
                new_data_df[statname].values[:] *= scale
                new_sigma_df[statname].values[:] *= scale

                # Remove duplicates
                duplicated = new_data_df.duplicated('DATE')
                new_data_df = new_data_df.loc[np.invert(duplicated),:]
                new_sigma_df = new_sigma_df.loc[np.invert(duplicated),:]

                # Append results
                data_df.append(new_data_df)
                sigma_df.append(new_sigma_df)
    
            # Reduce to make single data frame for current component
            data_df = reduce(lambda left,right: pd.merge(left, right, how='outer',
                             on='DATE'), data_df)
            sigma_df = reduce(lambda left,right: pd.merge(left, right, how='outer',
                              on='DATE'), sigma_df)

            # Set the DATE column to be the index in order to resample
            data_df.index = pd.to_datetime(data_df['DATE'], format='%Y-%m-%d %H:%M:%S.%f')
            sigma_df.index = pd.to_datetime(sigma_df['DATE'], format='%Y-%m-%d %H:%M:%S.%f')

            # Subset data by a time window
            ind = None
            if tstart is not None:
                ind  = data_df.index > np.datetime64(tstart)
            if tend is not None:
                ind *= data_df.index < np.datetime64(tend)
            if ind is not None:
                data_df = data_df[ind]
                sigma_df = sigma_df[ind]
 
            # Resample to an evenly spaced date range
            data_df = data_df.resample('D').mean().reset_index()
            sigma_df = sigma_df.resample('D').mean().reset_index()

            # Make date a separate column again
            data_df.rename(columns={'index': 'DATE'}, inplace=True)
            sigma_df.rename(columns={'index': 'DATE'}, inplace=True)

            # Save to table
            data_df.reset_index(drop=True).to_sql(component, engine_out.engine,
                                                  if_exists='replace')
            sigma_df.reset_index(drop=True).to_sql('sigma_' + component, engine_out.engine,
                                                   if_exists='replace')

        return


def save_block(data_df, sigma_df, engine, block_cnt):

    data_df = reduce(lambda left,right: pd.merge(left, right, how='outer',
        on='DATE'), data_df)
    sigma_df = reduce(lambda left,right: pd.merge(left, right, how='outer',
        on='DATE'), sigma_df)

    data_df.to_sql('data_block_%03d' % block_cnt, engine.engine, if_exists='replace')
    sigma_df.to_sql('sigma_block_%03d' % block_cnt, engine.engine, if_exists='replace')

    return [], [], block_cnt+1


def merge_blocks(block_cnt, comp, engine):

    for i in range(block_cnt):

        # If first block, save to new component
        if i == 0:
            df = pd.read_sql_table('data_block_%03d' % i, engine.engine)
            df.to_sql(comp, engine.engine, if_exists='replace')
            df = pd.read_sql_table('sigma_block_%03d' % i, engine.engine)
            df.to_sql('sigma_' + comp, engine.engine, if_exists='replace')

        # If other blocks, perform outer join
        else:
            cmd = ('SELECT * INTO %s FROM %s FULL OUTER JOIN data_block_%03d '
                   'ON %s.DATE = data_block_%03d.DATE;' % (comp, comp, i, comp, i))
            print(cmd)
            pd.io.sql.execute(cmd, engine.engine)


# end of file
