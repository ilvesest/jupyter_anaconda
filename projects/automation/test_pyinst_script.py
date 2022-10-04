#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 16:40:29 2022

@author: tonu

Testing 'pyinstaller' one click scheduling script
"""

import pandas as pd
from datetime import datetime

fpath = "/home/tonu/Documents/data_science/jupyter_anaconda/projects/automation/data"
df = pd.read_csv(f"{fpath}/auto.csv")

now = datetime.now()
year_month_day = now.strftime("%Y_%m_%d") #YYYY_MM_DD                 

df.head().to_csv(f"{fpath}/auto_{year_month_day}.csv")