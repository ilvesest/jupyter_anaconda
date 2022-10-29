#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:51:51 2022

@author: tonu
"""

import os
os.environ['KIVY_AUDIO'] = 'sdl2'

from kivy.config import Config
Config.set('graphics', 'width', '900')
Config.set('graphics', 'height', '400')



from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.relativelayout import RelativeLayout
from kivy.properties import NumericProperty, Clock, ObjectProperty, StringProperty
from kivy.graphics.context_instructions import Color
from kivy.graphics.vertex_instructions import Line, Quad, Triangle
from kivy.core.window import Window
from kivy import platform
from kivy.lang import Builder
from kivy.core.audio import SoundLoader

import random


#tell kivy to read helpers.menu.kv file
Builder.load_file("menu.kv")

class MainWidget(RelativeLayout):
    
    from helpers.transforms import transform, transform_2D, \
        transform_perspective
    from helpers.user_actions import keyboard_closed, on_keyboard_down, \
        on_keyboard_up, on_touch_down, on_touch_up
    
    menu_widget = ObjectProperty()
    perspective_point_x = NumericProperty(0)
    perspective_point_y = NumericProperty(0)
    
    V_LINES_N = 6
    V_LINES_SPACING = .4 # % in screen width 
    vertical_lines = []
    
    H_LINES_N = 15
    H_LINES_SPACING = .1 # % in screen height 
    horizontal_lines = []
    
    SPEED = .5
    current_offset_y = 0
    current_y_loop = 0
    
    SPEED_X = 3.0
    current_speed_x = 0
    current_offset_x = 0
    
    N_TILES = 12
    tiles = [] # list of Quads 
    tiles_coordinates = []
    
    SHIP_WIDTH = .1
    SHIP_HEIGHT = 0.035
    SHIP_BASE_Y = 0.04
    ship = None
    ship_coordinates = [(0, 0), (0, 0), (0, 0)]
    
    state_game_over = False
    state_game_has_started = False
    
    menu_title = StringProperty("G   A   L   A   X   Y")
    menu_button_title = StringProperty("START")
    score_txt = StringProperty()
    
    sound_begin = None
    sound_galaxy = None
    sound_gameover_impact = None
    sound_gameover_voice = None
    sound_music1 = None
    sound_restart = None
    
    def __init__(self, **kwargs):
        super(MainWidget, self).__init__(**kwargs)
        
        self.init_audio()
        self.init_vertical_lines()
        self.init_horizontal_lines()
        self.init_tiles()
        self.init_ship()
        self.pre_fill_tiles_coordinates()
        self.generate_tiles_coordinates()
        
        if self.is_desktop():
            self._keyboard = Window.request_keyboard(self.keyboard_closed,self)
            self._keyboard.bind(on_key_down=self.on_keyboard_down)
            self._keyboard.bind(on_key_up=self.on_keyboard_up)
        
        Clock.schedule_interval(self.update, 1.0 / 60.0)
        self.sound_galaxy.play()
            
        
    
    def init_audio(self):
        self.sound_begin = SoundLoader.load("audio/begin.wav")
        self.sound_galaxy = SoundLoader.load("audio/galaxy.wav")
        self.sound_gameover_impact = SoundLoader.load("audio/gameover_impact.wav")
        self.sound_gameover_voice = SoundLoader.load("audio/gameover_voice.wav")
        self.sound_music1 = SoundLoader.load("audio/music1.wav")
        self.sound_restart = SoundLoader.load("audio/restart.wav")
        
        self.sound_music1.volume = 1
        self.sound_galaxy.volume = .25
        self.sound_gameover_impact.volume = .6
        self.sound_gameover_voice.volume = .25
        self.sound_restart.volume = .25
        self.sound_begin.volume = .25
    
    
    def reset_game(self):
        #reset to inital state 
        self.current_offset_y = 0
        self.current_y_loop = 0
        self.current_speed_x = 0
        self.current_offset_x = 0
        
        #reset tile coordinates to start with straight line again
        self.tiles_coordinates = []
        self.score_txt = f"SCORE: {self.current_y_loop}"
        self.pre_fill_tiles_coordinates()
        self.generate_tiles_coordinates()
        
        self.state_game_over = False

    
    def is_desktop(self):
        if platform in ('linux', 'win', 'macosx'):
            return True
        return False
    
    
    def init_ship(self):
        with self.canvas:
            Color(0, 0, 0)
            self.ship = Triangle()
            
    
    def update_ship(self):
        
        ship_half_width = self.SHIP_WIDTH * self.width /2
        
        x2_ = self.width/2
        x1_ = x2_ - ship_half_width
        x3_ = x2_ + ship_half_width
        
        y1_ = self.SHIP_BASE_Y * self.height 
        y3_ = y1_
        y2_ = y1_ + self.SHIP_HEIGHT * self.height
        
        self.ship_coordinates[0] = (x1_, y1_)
        self.ship_coordinates[1] = (x2_, y2_)
        self.ship_coordinates[2] = (x3_, y3_)
        
        x1, y1 = self.transform(*self.ship_coordinates[0])
        x2, y2 = self.transform(*self.ship_coordinates[1])
        x3, y3 = self.transform(*self.ship_coordinates[2])
        
        self.ship.points = [x1, y1, x2, y2, x3, y3]
    
    
    def check_ship_collision(self):
        
        for i in range(len(self.tiles_coordinates)):
            
            ti_x, ti_y = self.tiles_coordinates[i]
            if ti_y > self.current_y_loop + 1: #cehck only first 2 tiles
                return False
            if self.check_ship_collision_with_tile(ti_x, ti_y):
                return True #collided
        return False #did not collide
    
    
    
    def check_ship_collision_with_tile(self, ti_x, ti_y):
        xmin, ymin = self.get_tile_coordinates(ti_x, ti_y)
        xmax, ymax = self.get_tile_coordinates(ti_x + 1, ti_y + 1)
        
        #loop through ship coordinate sand check if any of them are 
        #out of the allowed tile
        for i in range(3):
            px, py = self.ship_coordinates[i] #point x and y
            if xmin <= px <= xmax and ymin <= py <= ymax:
                return True
        return False #if no collision


    
    def init_tiles(self):    
        with self.canvas:
            Color(1, 1, 1) #white
            
            for i in range(self.N_TILES):
                self.tiles.append(Quad())
    
    
    def pre_fill_tiles_coordinates(self):
        self.tiles_coordinates = [(0, i) for i in range(10)]
    
    
    def generate_tiles_coordinates(self):
        
        last_x = 0
        last_y = 0
        
        start_line_x = -int(self.V_LINES_N/2) + 1
        end_line_x = start_line_x + self.V_LINES_N - 1
        
        rand_int = None
        
        #clean coordinates thself.init_ship()at are out of the screen
        #loop from the end of the tile coordinates
        for i in range(len(self.tiles_coordinates)-1, -1, -1):
            
            if self.tiles_coordinates[i][1] < self.current_y_loop:
                del self.tiles_coordinates[i]
        
        if len(self.tiles_coordinates) > 0:
            last_coordinates = self.tiles_coordinates[-1]
            last_x = last_coordinates[0]
            last_y = last_coordinates[1] + 1
        
        #fill tile coordinates to saisfy N_TILES requirement
        for i in range(len(self.tiles_coordinates), self.N_TILES):
            
            if last_x == start_line_x:
                rand_int = random.randint(0, 1)
            elif last_x == end_line_x - 1:
                rand_int = random.choice([0, 2]) 
            else:
                rand_int = random.randint(0, 2)
            # 0 -> stright #, 
            
                          #
            # 1 -> right ##, 
            
                        #
            # 2 -> left ##
            
            # forward tile always
            self.tiles_coordinates.append((last_x, last_y))
            
            if rand_int == 1:
                last_x += 1
                self.tiles_coordinates.append((last_x, last_y))
                last_y += 1
                self.tiles_coordinates.append((last_x, last_y))
                
            elif rand_int == 2:
                last_x -= 1
                self.tiles_coordinates.append((last_x, last_y))
                last_y += 1
                self.tiles_coordinates.append((last_x, last_y))
            
            last_y += 1
    
    
    def init_vertical_lines(self):    
        with self.canvas:
            Color(1, 1, 1) #white
            #self.line = Line(points=[100, 0, 100, 100])
            for i in range(self.V_LINES_N):
                self.vertical_lines.append(Line())
    
            
    def get_line_x_from_index(self, index):
        
        central_line_x = self.perspective_point_x
        spacing = self.V_LINES_SPACING * self.width
        offset = index - 0.5
        
        line_x = central_line_x + offset*spacing + self.current_offset_x
        return line_x
    
    
    def get_line_y_from_index(self, index):
        
        spacing_y = self.H_LINES_SPACING * self.height
        
        line_y = index*spacing_y - self.current_offset_y
        return line_y 
    
    
    def get_tile_coordinates(self, ti_x, ti_y):
        ti_y = ti_y - self.current_y_loop
        x = self.get_line_x_from_index(ti_x)
        y = self.get_line_y_from_index(ti_y)
        #print(f"x: {x}, y: {y}")
        return x, y
    
    

    def update_tiles(self):
        self.SHIP_WIDTH/2
        for i in range(self.N_TILES):
            ti_x = self.tiles_coordinates[i][0] 
            ti_y = self.tiles_coordinates[i][1]
            
            xmin, ymin = self.get_tile_coordinates(ti_x, ti_y)
            xmax, ymax = self.get_tile_coordinates(ti_x+1, ti_y+1)
            
            x1, y1 = self.transform(xmin, ymin)
            x2, y2 = self.transform(xmin, ymax)
            x3, y3 = self.transform(xmax, ymax)
            x4, y4 = self.transform(xmax, ymin)
            
            tile = self.tiles[i]
            tile.points = [x1, y1, x2, y2, x3, y3, x4, y4]


    def update_vertical_lines(self):

        start_index = -int(self.V_LINES_N/2) + 1
        for i in range(start_index, start_index + self.V_LINES_N):
            line_x = self.get_line_x_from_index(i)
            
            x1, y1 = self.transform(line_x, 0)
            x2, y2 = self.transform(line_x, self.height)
            
            self.vertical_lines[i].points = [x1, y1, x2, y2]
            
    
    def init_horizontal_lines(self):    
        with self.canvas:
            Color(1, 1, 1) #white
            for i in range(self.H_LINES_N):
                self.horizontal_lines.append(Line())
               
    
    def update_horizontal_lines(self):
        
        start_index = -int(self.V_LINES_N/2) + 1
        end_index = start_index + self.V_LINES_N - 1
        
        xmin = self.get_line_x_from_index(start_index)
        xmax = self.get_line_x_from_index(end_index)
        
        for i in range(0, self.H_LINES_N):
            line_y = self.get_line_y_from_index(i)
            
            x1, y1 = self.transform(xmin, line_y)
            x2, y2 = self.transform(xmax, line_y)
            
            self.horizontal_lines[i].points = [x1, y1, x2, y2]
    
    
    
    """Update clock schedule with delta time variable imitate forward 
    movement."""
    def update(self, dt):
        
        time_factor = dt*60
        
        # even though 'update' funtion should call itself 60 tps in reality
        # it will vary, since the game dynamics change becasue of that we need
        # to correct the progression of the game with the time variation 
        # 'time_factor'
        
        
        self.update_vertical_lines()
        self.update_horizontal_lines()
        self.update_tiles()
        self.update_ship()
        
        #moving forward only if not game over
        if not self.state_game_over and self.state_game_has_started:
            speed_y = self.SPEED * self.height / 100        
            self.current_offset_y += speed_y * time_factor
        
            spacing_y = self.H_LINES_SPACING * self.height
            #if the offset equals horizontal lien spacing reset to initial pos
            while self.current_offset_y >= spacing_y:
                self.current_offset_y -= spacing_y
                self.current_y_loop += 1
                self.score_txt = f"SCORE: {self.current_y_loop}"
                self.generate_tiles_coordinates()
        
            speed_x = self.current_speed_x * self.width / 100  
            self.current_offset_x += speed_x * time_factor    
        
        if not self.check_ship_collision() and not self.state_game_over:
            self.state_game_over = True
            self.menu_title = "G  A  M  E      O  V  E  R"
            self.menu_button_title = "RESTART"
            self.menu_widget.opacity = 1
            self.sound_music1.stop()
            self.sound_gameover_impact.play()
            Clock.schedule_once(self.play_game_over_voice_sound, 1)
            print("GAME OVER!")
    
    
    def play_game_over_voice_sound(self, dt):
        if self.state_game_over:
            self.sound_gameover_voice.play()
    
    
    def on_menu_button_pressed(self):
        
        if self.state_game_over:
            self.sound_restart.play()
        else:
            self.sound_begin.play()
        self.sound_music1.play()
            
        self.reset_game()
        self.state_game_has_started = True 
        self.menu_widget.opacity = 0
        
        
        
        
            
class GalaxyApp(App):
    pass


if __name__ == '__main__':
    GalaxyApp().run()