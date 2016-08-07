#-*- coding: utf-8 -*-

import numpy as np
import datetime as dtime
import pandas as pd
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
        for filepath in filelist:

            if obs_cnt % 100 == 0:
                sys.stdout.write(' - observation %d\r' % obs_cnt)
                sys.stdout.flush()

            # A little string parsing to get the station id
            statname = self.inst.parse_id(filepath)
            data['id'].append(statname)

            # Load all the data using numpy into an array of strings
            all_data = np.atleast_2d(np.loadtxt(filepath, dtype=bytes).astype(str))
            Nobs = all_data.shape[0]

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
                doy = all_data[:,cols['doy']].astype(int)
                for i in range(Nobs):
                    date = dtime.datetime(years[i], 1, 1)
                    data['DATE'].append(date + dtime.timedelta(doy[i]-1))

            # Save the file path
            self.engine.addFile(filepath)

            # Read meta data from header if we need to
            if read_header:
                self.inst.read_meta_header(filepath, meta_dict=meta_dict)

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


    def subset_table(self, idlist, engine_out):
        """
        Subset a raw data table using a list of station IDs, and perform
        an outer join using the dates.
        """
        # Loop over the components
        query = "SELECT DATE, %s, sigma_%s FROM tseries WHERE id = '%s';"
        for component in self.inst.components:

            # Loop over the stations
            data_df = None; sigma_df = None
            for i, statname in enumerate(idlist):

                # Query database for current station+component 
                data = pd.read_sql_query(query % (component, component, statname),
                    self.engine.engine)
                new_data_df = data[['DATE', component]]
                new_sigma_df = data[['DATE', 'sigma_' + component]]

                # Use station name in columns as a unique identifier
                new_data_df.columns = ['DATE', statname]
                new_sigma_df.columns = ['DATE', statname]

                # Merge results
                if data_df is None:
                    data_df, sigma_df = new_data_df, new_sigma_df
                else:
                    data_df = pd.merge(data_df, new_data_df, how='outer', on='DATE')
                    sigma_df = pd.merge(sigma_df, new_sigma_df, how='outer', on='DATE')

            # Set the DATE column to be the index in order to resample
            data_df.index = pd.to_datetime(data_df['DATE'], format='%Y-%m-%d %H:%M:%S.%f')
            sigma_df.index = pd.to_datetime(sigma_df['DATE'], format='%Y-%m-%d %H:%M:%S.%f')

            # Resample to an evenly spaced date range
            data_df = data_df.resample('D').sum().reset_index()
            sigma_df = sigma_df.resample('D').sum().reset_index()

            # Make date a separate column again
            data_df.rename(columns={'index': 'DATE'}, inplace=True)
            sigma_df.rename(columns={'index': 'DATE'}, inplace=True)

            # Save to table
            data_df.reset_index(drop=True).to_sql(component, engine_out.engine,
                if_exists='replace')
            sigma_df.reset_index(drop=True).to_sql('sigma_' + component, engine_out.engine,
                if_exists='replace')

        return


# end of file