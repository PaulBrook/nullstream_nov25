#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:53:42 2019

@author: jgoldstein
"""
import sys
print('python version: {}'.format(sys.version))

try:
    import astropy
    print(astropy)
    print(astropy.__version__)
except:
    print('no astropy!')
    
try:
    import cycler
    print(cycler)
    print(cycler.__version__)
except:
    print('no cycler!')
    
try:
    import kiwisolver
    print(kiwisolver)
    print(kiwisolver.__version__)
except:
    print('no kiwisolver!')
    
try:
    import matplotlib
    print(matplotlib)
    print(matplotlib.__version__)
except:
    print('no matplotlib!')
    
try:
    import numpy
    print(numpy)
    print(numpy.__version__)
except:
    print('no numpy!')
    
try:
    import pandas
    print(pandas)
    print(pandas.__version__)
except:
    print('no pandas!')
    
try:
    import pyparsing
    print(pyparsing)
    print(pyparsing.__version__)
except:
    print('no pyparsing!')
    
try:
    import dateutil
    print(dateutil)
    print(dateutil.__version__)
except:
    print('no dateutil!')

try:
    import pytz
    print(pytz)
    print(pytz.__version__)
except:
    print('no pytz')
    
try:
    import scipy
    print(scipy)
    print(scipy.__version__)
except:
    print('no scipy')
    
try:
    import six
    print(six)
    print(six.__version__)
except:
    print('no six')
