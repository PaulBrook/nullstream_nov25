#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 13:38:08 2020

@author: jgoldstein

some utility functions for all the analysis and plotting scripts
"""

import pandas as pd

def get_post(file_path):
    post = pd.read_csv(file_path, delim_whitespace=True, header=None, comment='#')
    # read in column names from first line
    # (can't read them in with pandas bc they're commented with #)
    with open(file_path) as f:
        first_line = f.readline()
    cols = first_line.split()[1:]
    post.columns = cols
    return post

