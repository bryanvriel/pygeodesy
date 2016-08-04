#-*- coding: utf-8 -*-

from __future__ import print_function
import sys


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('model', parent_package, top_path)
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)

# end of file
