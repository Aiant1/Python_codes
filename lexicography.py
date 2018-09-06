# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 19:24:06 2018

@author: Antika
"""

n=int(input("enter the number of skatingtrading cards"))   
list_of_cards=[]       
for i in range(0, n): # set up loop to run 5 times
	number = str(input('Please enter the card names: ')) # prompt user for number
	list_of_cards.append(number) # append to our_list
print ("\n")
count =0
swaped = True #Just to enter the first time
while swaped:
    swaped = False
    for i in range(len(list_of_cards)-1):
        if list_of_cards[i] > list_of_cards[i+1]:
            aux = list_of_cards[i]
            list_of_cards[i] = list_of_cards[i+1]
            list_of_cards[i+1] = aux
            swaped = True
            count = count+1
print("list_of_cards lexicographically",list_of_cards)
print ("\n")
#multiply count with $1
print("total cost moist have to pay $",count*1)