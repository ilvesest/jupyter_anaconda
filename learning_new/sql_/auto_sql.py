#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 22:09:33 2019

@author: ilves
"""

import sqlalchemy as db
    
def column_names(table_name, engine):
    ''' Function to print sqlalchemy Table object column
        names into the screen from a database (DB).
        
        Parameters:
            table_name : name of the table in the DB [str]
            engine : engine object
            
        Returns: Nothing'''
    
    # creates a Table and MetData objects
    table_object = db.Table(table_name, 
                            db.MetaData(), 
                            autoload=True, 
                            autoload_with=engine)
    
    print(table_object.columns.keys())