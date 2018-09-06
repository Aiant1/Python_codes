# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 15:00:29 2018

@author: ASUS
"""
import sys
import math
choco=[]
#enter the number of collegues
number_of_collegues= int(input("enter the number of collegues"))

#enter the number of chocolates each colleque has and append them in a list
for x in range(number_of_collegues):
    y = int(input("enter the number choco each collegue has"))
    
    choco.append(y)

choco.sort()

#limits the size of Python's data structures such as strings and lists.
sum1 = sys.maxsize
#base values(0,1,2)
for j in range (0,3):
    
    current_sum = 0
#optimum number of solutions
    for i in range(len(choco)): 
        delta = choco[i] - choco[0] + j
        current_sum = current_sum + int (delta / 5 + delta % 5 / 2 + delta % 5 % 2 / 1)
    
    sum1 = min(current_sum,sum1)
print ("\n")
print ("The optimal number of solution is = ",sum1)