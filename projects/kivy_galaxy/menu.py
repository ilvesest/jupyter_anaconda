#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 12:22:48 2022

@author: tonu
"""

from kivy.uix.relativelayout import RelativeLayout

class MenuWidget(RelativeLayout):
    
    """Define function to move the spaceship around on a touchscreen device."""
    def on_touch_down(self, touch):
        
        if self.opacity == 0:
            return False
        
        return super(RelativeLayout, self).on_touch_down(touch)
            