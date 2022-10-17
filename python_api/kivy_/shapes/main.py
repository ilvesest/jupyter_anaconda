#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 19:11:08 2022

@author: tonu
"""

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics.vertex_instructions import Line
from kivy.graphics.context_instructions import Color
from kivy.graphics.vertex_instructions import Rectangle

#filled rectangle, circle, line
class Canvas1(Widget):
    pass

#center positioned cross
class Canvas2(Widget):
    pass

#outlined and colored rectangle, circle, line
class Canvas3(Widget):
    pass


class Canvas4(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            Line(points=(100, 100, 400, 500), width=2)
            Color(0, 1, 0)
            Line(circle=(400, 200, 80), width=2)
            Line(rectangle=(500, 250, 100, 150), width=4)
            Color(0, 0, 1)
            Rectangle(pos=(700, 200), size=(150, 100))

class ShapesApp(App):
    pass



if __name__ == "__main__":
    ShapesApp().run()