#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 21:31:48 2022

@author: tonu
"""

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.pagelayout import PageLayout
from kivy.uix.button import Button
from kivy.metrics import dp


class BoxLayoutSamples(BoxLayout):
    pass

class AnchorLayoutSamples(AnchorLayout):
    pass

class GridLayoutSamples(GridLayout):
    pass

class StackLayoutSamples(StackLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        for i in range(100):
# =============================================================================
#             b = Button(text=str(i), size_hint=(.2, .2))
#             self.add_widget(b)
# =============================================================================
            b = Button(
                text=str(i), 
                size_hint=(None, None), 
                size=(dp(100), dp(100)))
            self.add_widget(b)
            
    pass

class ScrollViewSamples(ScrollView):
    pass

class PageLayoutSamples(PageLayout):
    pass

class MainWidget(Widget):
    pass

class LayoutsApp(App):
    pass

if __name__ == "__main__":
    LayoutsApp().run()