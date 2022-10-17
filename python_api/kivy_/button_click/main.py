#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 08:48:21 2022

@author: tonu
"""

from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.properties import StringProperty, BooleanProperty


class ButtonWidgets(GridLayout):    
        
    ### TOGGLE, BUTTON & LABEL ###

    counter = 0
    counter_text = StringProperty('0')
    toggle_enabled = BooleanProperty(False)
        
    label_text = StringProperty("Hello!")
    def on_button_click(self):
        print('Button clicked') #this is outputted in trace only not on GUI
        self.label_text = "Successful!"
    
    
    def on_button_click_count(self):
        if self.toggle_enabled:
            self.counter += 1
            self.counter_text = str(self.counter)
         
    #have to use widget parameter to reference the 'self' in .kv file
    #the first 'self' here references to the python class (ButtonWidgets) instance
    def on_toggle_button_state(self, widget):
        print(f"toggle state: {widget.state}")
        if widget.state == 'normal':
            widget.text = 'OFF'
            self.toggle_enabled = False
        else:
            widget.text = "ON"
            self.toggle_enabled = True
    
    
    ### SWITCH, SLIDER & PROGRESS BAR ###
    
    """Binding switch with slider through 'id' property of Switch in .kv.
    Init Switch and Slider to be active initially through respective'active' 
    and 'disabled' properties."""
    
    text_input_str = StringProperty("John Doe")
    
    def on_text_validate(self, widget):
        self.text_input_str = widget.text
    
class WidgetsApp(App):
    pass


if __name__ == "__main__":
    WidgetsApp().run()
    
    
