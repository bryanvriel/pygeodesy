#-*- coding: utf-8 -*-

from __future__ import print_function
import configparser
import sys


class Configuration:
    """
    Reads in a configuration file for processing options.
    """

    def __init__(self, cfgfile, module):
        """
        Store the filename of the configuration file and the module to configure.
        """
        self.cfgfile = cfgfile
        self.module = module
        return


    def __call__(self):
        """
        Parse configuration in cfgfile for a given module.
        """

        # Create configparser object
        config = configparser.RawConfigParser(inline_comment_prefixes=('#',';'))
        config.read(self.cfgfile)

        # Check module is listed in one of the sections
        if self.module not in config.sections():
            print('No configuration information found for %s' % self.module)
            print('Found config sections', config.sections())
            assert False, 'Configuration error'

        # Get global options
        optdict = {key: value for (key,value) in config.items('global')}

        # Get options for module
        for key,value in config.items(self.module):
            
            # Check if boolean
            if value in ('True', 'true'):
                optdict[key] = True
            elif value in ('False', 'false'):
                optdict[key] = False
            else:
                optdict[key] = value

        return optdict

# end of file
