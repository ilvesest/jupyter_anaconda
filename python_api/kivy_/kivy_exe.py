#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 08:00:56 2022

@author: tonu
@title: Kivy API for creating executable files
"""

# =============================================================================
# Pyinstaller hooks. 
# Source: https://stackoverflow.com/questions/37696206/how-to-get-an-windows-executable-from-my-kivy-app-pyinstaller
# 
# =============================================================================
from kivy.tools.packaging import pyinstaller_hooks as hooks

kivy_deps_all = hooks.get_deps_all() # dct('datas', 'hiddenimports', 'excludes')