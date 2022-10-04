#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 19:02:50 2022

@author: tonu

title: Scheduling Scripts demo
"""

from selenium import webdriver
from selenium.webdriver.firefox.options import Options #for headless mode
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager

from datetime import datetime
import os, sys
import pandas as pd

options = Options()
options.headless = True #headless mode parameter

driver = webdriver.Firefox(
    service=FirefoxService(GeckoDriverManager().install()), 
    options=options)

url = "https://www.thesun.co.uk/sport/football/"
driver.get(url)

#find all football news containers
containers = driver.find_elements(
    by='xpath', 
    value='//div[@class="teaser__copy-container"]'
    )

#find title, subtitles and hyperlink texts
titles, sub_ts, hrefs = [], [], []
for i,container in enumerate(containers):
    
    if i == 5: break
    
    titles.append(container.find_element(by='xpath', value='./a/h2').text)
    sub_ts.append(container.find_element(by='xpath', value='./a/p').text)
    hrefs.append(container.find_element(by='xpath', value='./a').get_attribute("href"))

driver.quit()
    
df = pd.DataFrame(data=
                  {'titles':titles,
                   'sub_titles': sub_ts,
                   'hrefs':hrefs})

"""Write df to csv with timestamp"""

#app_path = os.path.dirname(sys.executable)
app_path = os.getcwd()

now = datetime.now()
year_month_day = now.strftime("%Y_%m_%d") #YYYY_MM_DD

file_name = f"headlines-{year_month_day}.csv"

#for different OS compability use OS module to generate the path
#since macOS, linux and windows have different path methods
final_path = os.path.join(app_path, "tests" ,file_name)

df.to_csv(final_path)
