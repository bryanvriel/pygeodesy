#-*- coding: utf-8 -*-

# The submodules
from . import db
from . import network
from . import instrument
from . import model
from . import view

# Make action public
from .components import action

def main():
    """
    The main entrypoint to pygeodesy using the plexus.
    """
    return pygeodesy.run()

def boot():
    """
    Internal function to create the plexus and initialize the dashboard. Used the
    merlin package as a template.
    """
    # Access the plexus factory
    from .components import pygeodesy
    # Build one
    plexus = pygeodesy(name='pygeodesy.plexus')

    # Get the dashboard
    from .components import dashboard
    # Attach the singletons
    import weakref
    dashboard.pygeodesy = weakref.proxy(plexus)

    return plexus

# Call boot() to get a pygeodesy plexus
pygeodesy = boot()

# Meta information
from . import meta

# end of file
