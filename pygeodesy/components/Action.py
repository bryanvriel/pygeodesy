#-*- coding: utf-8 -*-

import pyre
from .Dashboard import Dashboard

class Action(pyre.action, Dashboard, family='pygeodesy.tasks'):
    """
    Protocol for pygeodesy commands.
    """

# end of file
