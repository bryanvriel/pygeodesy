#-*- coding: utf-8 -*-

import pyre
import pygeodesy as pg

@pyre.foundry(implements=pg.action, tip='Make a time series database')
def makedb():
    from .MakeDB import MakeDB
    return MakeDB

@pyre.foundry(implements=pg.action, tip='Subset a time series database file')
def subnet():
    from .Subnet import Subnet
    return Subnet

@pyre.foundry(implements=pg.action,
              tip='Clean a time series by removing outliers and removing bad values')
def clean():
    from .Clean import Clean
    return Clean

@pyre.foundry(implements=pg.action, tip='Apply low-pass/smoothing filter to time series database')
def filter():
    from .Filter import Filter
    return Filter

@pyre.foundry(implements=pg.action, tip='Perform common mode estimation on data')
def cme():
    from .CommonModeEstimation import CommonModeEstimation
    return CommonModeEstimation

@pyre.foundry(implements=pg.action, tip='Fit temporal model to time series database')
def modelfit():
    from .ModelFit import ModelFit
    return ModelFit

@pyre.foundry(implements=pg.action, tip='Remove time series model from time series data')
def detrend():
    from .Detrend import Detrend
    return Detrend

@pyre.foundry(implements=pg.action,
              tip='Fit temporal model to time series database using Elastic Net regression')
def elasticnet():
    from .ElasticNet import ElasticNet
    return ElasticNet

@pyre.foundry(implements=pg.action, tip='Plot component time series for specified stations')
def plot():
    from .Plot import Plot
    return Plot

@pyre.foundry(implements=pg.action, tip='Makes interactive map of station network')
def netmap():
    from .NetMap import NetMap
    return NetMap

@pyre.foundry(implements=pg.action, tip='Plot vector map of displacements for network')
def velmap():
    from .VelMap import VelMap
    return VelMap

# end of file
