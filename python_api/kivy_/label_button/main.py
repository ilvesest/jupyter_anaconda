# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 15:39:11 2022

@author: tonu
"""

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button


class BoxLayoutSample(BoxLayout):
    pass
    #create box layout through python code
# =============================================================================
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         button1 = Button(text="A")
#         button2 = Button(text="B")
#         self.add_widget(button1)
#         self.add_widget(button2)
# =============================================================================

#create some buttons and labels through kivy code inside .kv file
#add MainWidget at the beginning
class MainWidget(Widget):
    pass

class LabelButtonApp(App):
    pass

if __name__ == "__main__":
    LabelButtonApp().run()