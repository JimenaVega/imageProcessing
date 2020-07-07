# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:43:28 2020

@author: miner
"""

from tool._fixedInt import *

a=DeFixedInt(8,2,'S','round','saturate')

a.value = -1

print ("\nPara a: ")
print ('Float: %f|'%a.fValue,'NBI: %d|'%a.intWidth,'NBF: %d|'%a.fractWidth,'NB: %d|'%a.width)
print ("\nEntero Equivalente")
print ('Int: %d|'%a.intvalue,'Bin: ',bin(a.intvalue))
print ("\n----> Rango de a: ",a.showRange())



a.value = 8

print ("\nPara b: ")
print ('Float: %f|'%a.fValue,'NBI: %d|'%a.intWidth,'NBF: %d|'%a.fractWidth,'NB: %d|'%a.width)
print ("\nEntero Equivalente")
print ('Int: %d|'%a.intvalue,'Bin: ',bin(a.intvalue))
print ("\n----> Rango de a: ",a.showRange())