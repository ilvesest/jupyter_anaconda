# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a file where I (Tonu) have stored some of my own functions. 
"""

#imports:


#functions
def to_file(filename, data, extension=''):
    """Writes data to a file.
    
    Parameters:
        filename (str): name of the file
        data (str): data to be written on the file
        
    Keyword arguments:
        extension (str): file extension with the dot (default '')
        
    Returns: None
    """

    with open(filename + extension, 'w') as file:
        file.write(data)
        
    print('File "{}{}" created.'.format(filename, extension))

### ### ###

def si_prefix(number, unit):
    """Convert a value into specified (SI system) prefix-unit representation.
    
    Parameters:
        number (int/float): number to be converted 
        unit (str): physical unit to be used
        
    Returns: (str)
    """
    
    import math
    
    n = float(abs(number))
    
    if abs(number) > 0 and abs(number) < 1:
        neg_symbols = ['','m','μ','n','p']
        
        #finding appropriate negative prefix index
        i = max(0, min(len(neg_symbols)-1, 
                   int(abs(math.floor(math.log10(n) / 3)))))
        
        return '{:.2f} {}'.format(number * 10 ** (i * 3), neg_symbols[i]+unit)
    
    else:
        pos_symbols = ['','k','M','G','T']
        
        #finding appropriate positive prefix index
        i = max(0, min(len(pos_symbols)-1, 
                   int(math.floor(math.log10(abs(n)) / 3))))
        
        return '{:.2f} {}'.format(number / 10 ** (i * 3), pos_symbols[i]+unit)
    
### ### ###
        
def all_errors():
    """Returns all possible Python errors.
    
    Returns: (list)
    """
    
    import re
    
    return sorted([x for x in __builtins__.__dict__.keys() if re.compile(r"^.*Error").search(x)])