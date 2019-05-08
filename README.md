## pygeodesy configuration

In this branch, `pygeodesy` uses the `pyre` package (https://github.com/pyre/pyre) to handle configuration and documentation of parameters. Similar to the previous version of `pygeodesy`, all tasks flow through a single application binary called `pygeodesy`.
```

> pygeodesy --help

pygeodesy: 

    pygeodesy 0.0.1
    copyright (c) 2016-2019 all rights reserved

Author:
    Bryan V. Riel <bryanvriel@gmail.com>

commands:
        makedb: Make a time series database
        subnet: Subset a time series database file
         clean: Clean a time series by removing outliers and removing bad values
        filter: Apply low-pass/smoothing filter to time series database
           cme: Perform common mode estimation on data
      modelfit: Fit temporal model to time series database
       detrend: Remove time series model from time series data
    elasticnet: Fit temporal model to time series database using Elastic Net regression
          plot: Plot component time series for specified stations
        netmap: Makes interactive map of station network
        velmap: Plot vector map of displacements for network
        
```

Each task handled by `pygeodesy` also comes with its own help explaining all its parameters:

```

> pygeodesy modelfit --help

pygeodesy: 

    pygeodesy 0.0.1
    copyright (c) 2016-2019 all rights reserved

Author:
    Bryan V. Riel <bryanvriel@gmail.com>
pygeodesy modelfit

    Fit temporal model to time series database.

usage:
    pygeodesy modelfit [command]

where [command] is one of
    help: show this help screen

options:
            --input: Input time series database [str]
           --output: Output time series model database [str]
             --user: Python file defining temporal dictionary [str]
             --nstd: Number of deviations for outlier threshold (default: 5) [int]
               --t0: Starting decimal year of model (default: None) [float]
               --tf: Ending decimal year of model (default: None) [float]
          --penalty: Regularization parameter for model estimator (default: 1.0) [float]
       --output_amp: Filename for optionally saving seasonal amplitudes for each station [str]
     --output_phase: Filename for optionally saving seasonal phase for each station [str]
         --num_iter: Number of iterations for iterative least squares (default: 1) [int]
          --rw_iter: Number of re-weighting iterations for LassoRegression (default: 1) [int]
           --solver: Name of solver from [RidgeRegression, LassoRegression] [str]
            --scale: Scale observations by factor (default: 1.0) [float]
    --special_stats: List of stations to allow for 10x higher std threshold [str]
       --std_thresh: Absolute deviation threshold for bad stations (default: 1.0e10) [float]
     --min_timespan: Minimum timespan (days) of valid data to keep station (default: 365.0) [float]
        --min_valid: Minimum number of valid observations to keep station (default: 100) [int]
              --dry: show what would get done without actually doing anything [bool]
              
```

Also similar to the previous version of `pygeodesy`, all parameters for all tasks can be stored in a single configuration file. Here, we use the `pfg` file format which uses whitespace to demarcate different sections of the `pfg` file for different tasks. For example, here is a `pfg` file that specifies parameters for the tasks `makedb` and `subnet`:
```

;
; Configuration file for GPS data
;

pygeodesy:

    ; Global configuration
    global:
        ; The type of time series data
        data_type = gps
        ; Supported data format?
        data_format = sopac

    ; Make database from time series files
    makedb:
        ; Input time series directory
        directory = /Users/briel/data/WNAM
        ; The column format of files
        column_fmt = year: 1, doy: 2, north: 3, east: 4, up: 5, sigma_north: 6, sigma_east: 7, sigma_up: 8
        ; Output database name
        dbname = raw.db
        ; File containing coordinates of stations
        metafile = /Users/briel/data/SopacStack/SopacCoordinates.txt
        ; The relevant columns in metafile
        metafmt = id: 0, lat: 8, lon: 9, elev: 10
        ; The file extension for time series files
        extension = .neu

    ; Subset database for a given spatial region and temporal window
    subnet:
        ; Input raw database
        input = sqlite:///raw.db
        ; Output subset database
        output = sqlite:///data.db
        ; Polygon coordinates file
        poly = poly_slab.txt
        ; Start time
        tstart = 2010-01-01
        ; End time
        tend = 2019-01-01
        ;station_list = cascadia_stations.txt
        
```

If you name your parameter file `pygeodesy.pfg`, then the `pygeodesy` application will find the `pfg` file automatically. If you name your `pfg` file something else, you just need to pass the name of the file via the `--config` flag, e.g.:

```

> pygeodesy makedb --config=myfile.pfg

```

Parameters can also be specified on the command line to override the parameters in the `.pfg` file, e.g.:

```

> pygeodesy modelfit --std_thresh=100.0

```