#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 08:38:07 2022

@author: tonu
"""
"""IMPORTS"""

import kivy
from kivy.app import App
from kivy.uix.label import Label

#resize the app window from fullscreen to smaller
from kivy.config import Config
Config.set('graphics', 'fullscreen', '0')



"""ADD LABEL"""
class MyApp(App):
    
    def build(self):
        return Label(text="Hello World!", font_size=52)
    
if __name__ == '__main__':
    MyApp().run()