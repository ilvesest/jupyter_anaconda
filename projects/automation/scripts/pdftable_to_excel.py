#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 20:32:39 2022

@author: tonu

Topic: Script to save a table from .pdf file downladed on webpage.
"""

#imports
import requests
import camelot

web_url = "https://camelot-py.readthedocs.io/en/master/_static/pdf/foo.pdf"
filename = "file1.pdf"

#downloads sample pdf file to current directory
def download_pdf(url, file_name=filename):
    response_obj = requests.get(url)
    
    response_status_code = response_obj.status_code
    
    #assert response was succesful (status code = 200)
    if response_status_code == requests.codes.ok:
        print("PDF obtained.")
        with open(file_name, 'wb') as file:
            file.write(response_obj.content)
        print(f"PDF saved as '{file_name}'") 
    else:
        raise ValueError(f"Response status code: {response_status_code}")

#extracts table from pdf as df
def extract_pdf_tables(file_path):
    pdf_tables = camelot.read_pdf(
        filepath=file_path,
        pages='all',
        flavor='lattice',
        backend='poppler'
    )
    
    return pdf_tables[0].df

if __name__ == '__main__':
    
    #download pdf
    download_pdf(web_url)
    
    #extract table as df
    df = extract_pdf_tables(filename)
    
    #save table to xlsx
    df.to_excel('table1.xlsx', header=False, index=False)