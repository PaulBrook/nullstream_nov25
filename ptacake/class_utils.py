#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 11:37:31 2019

@author: jgoldstein
"""

# decorator for a class
# returns the same class but with the functions in functions added as methods
def add_functions_as_methods(functions):
    def decorator(Class):
        for function in functions:
            setattr(Class, function.__name__, function)
        return Class
    return decorator