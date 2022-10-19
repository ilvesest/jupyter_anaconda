#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 19:11:08 2022

@author: tonu
"""

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics.vertex_instructions import Line, Rectangle, Ellipse
from kivy.metrics import dp
from kivy.graphics.context_instructions import Color
from kivy.properties import Clock
from kivy.uix.boxlayout import BoxLayout

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
            self.rect = Rectangle(pos=(200, 200), size=(150, 100))
            
    def on_button_click(self):
        w, h = self.rect.size
        x1, y1 = self.rect.pos
        inc = dp(30) #increment
        
        #distance to the window wall to the right
        diff_right = self.width - (x1 + w)
        
        if diff_right <= inc:
            self.rect.pos = (self.width-w, y1)
        else:
            self.rect.pos = (x1+inc, y1)

        
class Canvas5(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.ball_size = dp(50)
        self.v_x = dp(3) #ball speed in x
        self.v_y = dp(4) #ball speed in y
        with self.canvas:
            """Ball is drawn before canvas, we cannot center it when we init 
            it. But we can create a function after init, then cnvas size is 
            known"""
            self.ball = Ellipse(
                pos=(100, 100), 
                size=(self.ball_size, self.ball_size))    
        
        #update ball position every x seconds
        Clock.schedule_interval(self.update, 1/60)
        
    def on_size(self, *args):
        #print(f"Window size: {self.width},{self.height}")
        
        self.ball.pos = (self.center_x - self.ball_size/2,
                         self.center_y - self.ball_size/2)
    
    #function that updates clock must have 'dt' argument
    def update(self, dt):
        
        x1, y1 = self.ball.pos #ball bottom left coordinates
        x2, y2 = x1 + self.ball_size, y1 + self.ball_size #top right
        
        if y2 > self.height:
            y1 = self.height - self.ball_size
            self.v_y = -self.v_y
        if x2 > self.width:
            x1 = self.width - self.ball_size
            self.v_x = -self.v_x
        if y1 < 0:
            y1 = 0
            self.v_y = -self.v_y
        if x1 < 0:
            x1 = 0
            self.v_x = -self.v_x
        
        self.ball.pos = (x1 + self.v_x, y1 + self.v_y)

# draw french flag using relative layout in box layout
class Canvas6(BoxLayout):
    pass
        
class ShapesApp(App):
    pass



if __name__ == "__main__":
    ShapesApp().run()