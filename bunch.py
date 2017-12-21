# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 10:47:05 2017

@author: User
"""

class Bunch(dict):
    def __init__(self, *args, **kwds):
        super(Bunch,self).__init__(*args, **kwds)
        self.__dict__ = self
        
