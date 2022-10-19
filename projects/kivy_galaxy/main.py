#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:51:51 2022

@author: tonu
"""

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import NumericProperty
from kivy.graphics.context_instructions import Color
from kivy.graphics.vertex_instructions import Line


class MainWidget(Widget):
    
    perspective_point_x = NumericProperty(0)
    perspective_point_y = NumericProperty(0)
    
    V_LINES_N = 15
    V_LINES_SPACING = .25 # % in screen width 
    vertical_lines = []
    
    H_LINES_N = 10
    H_LINES_SPACING = .1 # % in screen height 
    horizontal_lines = []
    
    def __init__(self, **kwargs):
        super(MainWidget, self).__init__(**kwargs)
        #print(f"INIT W: {self.width} H: {self.height}")
        self.init_vertical_lines()
        self.init_horizontal_lines()
        
    def on_parent(self, widget, parent):
        #print(f"ON PARENT W: {self.width} H: {self.height}")
        pass
    
    def on_size(self, *args):
        # print(f"ON SIZE W: {self.width} H: {self.height}")

        #self.perspective_point_x = self.width / 2
        #self.perspective_point_y = self.height * 0.75
        self.update_vertical_lines()
        self.update_horizontal_lines()
        
        pass
    
    def on_perspective_point_x(self, widget, value):
        # print(f"PX: {value}")
        pass

    def on_perspective_point_y(self, widget, value):
        # print(f"PY: {value}") 
        pass

    def init_vertical_lines(self):    
        with self.canvas:
            Color(1, 1, 1) #white
            #self.line = Line(points=[100, 0, 100, 100])
            for i in range(self.V_LINES_N):
                self.vertical_lines.append(Line())
            
            
    def update_vertical_lines(self):
        
        central_line_x = int(self.width / 2)
        spacing = self.V_LINES_SPACING * self.width
        offset = -int(self.V_LINES_N/2) + 0.5 # = 3
        
        for i in range(self.V_LINES_N):
            line_x = int(central_line_x + offset * spacing)
            
            x1, y1 = self.transform(line_x, 0)
            x2, y2 = self.transform(line_x, self.height)
            
            self.vertical_lines[i].points = [x1, y1, x2, y2]
            offset += 1
            
    
    def init_horizontal_lines(self):    
        with self.canvas:
            Color(1, 1, 1) #white
            for i in range(self.H_LINES_N):
                self.horizontal_lines.append(Line())
            
            
    def update_horizontal_lines(self):
        
        central_line_x = int(self.width / 2)
        spacing = self.V_LINES_SPACING * self.width
        offset = -int(self.V_LINES_N/2) + 0.5 # = 3
        
        xmin = central_line_x + offset * spacing
        xmax = central_line_x - offset * spacing
        spacing_y = self.H_LINES_SPACING * self.height
        
        for i in range(self.H_LINES_N):6
            line_y = i * spacing_y
            
            x1, y1 = self.transform(xmin, line_y)
            x2, y2 = self.transform(xmax, line_y)
            
            self.horizontal_lines[i].points = [x1, y1, x2, y2]
            
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
        
        x_t = self.perspective_point_x + (x_diff * factor_y)
        y_t = (1 - factor_y) * self.perspective_point_y
        
        return int(x_t), int(y_t)
    
class GalaxyApp(App):
    pass


if __name__ == '__main__':
    GalaxyApp().run()