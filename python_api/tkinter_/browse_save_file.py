#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 17:18:04 2022

@author: tonu
"""
import pandas as pd

import tkinter as tk
from tkinter import filedialog

# tkinter OS native file manager to save DataFrame to file
def tk_browse_to_save_df(
        df: pd.DataFrame,
        default_filename: str = "Dataframe_1",
        default_extension: str = ".csv",
        filetypes: list(tuple) = [('csv file', '.csv')]):
    """
    Use OS native browse window with tkinter to save df to file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be saved.
    default_filename : str, optional
        Default filename in browse window for the DF. The dfault is
        "Dataframe_1".
    filetypes : list(tuple), optional
        List of ("filetype", ".extension") tuples for file manager to show in
        a given directory. The default is [('csv file', '.csv')].
    defaultextension : str, optional
        Default file extension to be saved to. The default is ".csv".

    Returns
    -------
    None.

    """
    # open OS native file browser with tkinter
    root = tk.Tk()   # toplevel widget/window
    root.withdraw()  # hide toplevel window

    # configure the file parameters
    file = filedialog.asksaveasfilename(
        filetypes=filetypes,
        defaultextension=default_extension,
        initialfilename=default_filename)

    if file and default_extension == '.csv':
        df.to_csv(file, index=False)
    elif file and default_extension in ['.xlsx', '.odt']:
        df.to_excel(file, index=False)