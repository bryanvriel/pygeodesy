#-*- coding: utf-8 -*-

from __future__ import print_function
import sys


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('pygeodesy', parent_package, top_path)
    config.add_subpackage('db')
    config.add_subpackage('network')
    config.add_subpackage('instrument')
    config.add_subpackage('model')
    config.add_subpackage('view')
    config.add_subpackage('components')
    config.add_subpackage('tasks')
    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)

# end of file
