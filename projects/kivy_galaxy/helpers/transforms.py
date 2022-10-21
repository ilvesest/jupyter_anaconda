#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 21:43:35 2022

@author: tonu
"""

"""Transforming from 2D view to perspective view"""
def transform(self, x, y):
    #return self.transform_2D(x, y) # logic is done in 2D
    return self.transform_perspective(x, y)

def transform_2D(self, x , y):
    return int(x), int(y)

def transform_perspective(self, x , y):
    
    #transform y coordinate of the line
    y_lin = y * self.perspective_point_y / self.height
    if y_lin > self.perspective_point_y:
        y_lin = self.perspective_point_y
    
    x_diff = x - self.perspective_point_x
    y_diff = self.perspective_point_y - y_lin
    factor_y = y_diff / self.perspective_point_y
    factor_y = factor_y ** 4 # to emphasize 3D view
    
    x_t = self.perspective_point_x + x_diff * factor_y
    y_t = (1 - factor_y) * self.perspective_point_y
    
    return int(x_t), int(y_t)