#-*- coding: utf-8 -*-

class GenericClass:
    pass

import sys
majorVersion = sys.version_info[0]

from . import network
from . import db
from . import instrument
from . import model

# end of file
