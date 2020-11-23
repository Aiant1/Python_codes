# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 18:09:59 2020

@author: Antika Das
"""
import time
import sys
from threading import Thread
#from time import sleep
l=[]
Validated_coins=[10,20,50,100,200,500]
Validated_coins.reverse()
print (Validated_coins)
COCA=260
FANTA=240 
HOT_CHOCOLATE=350
LATTE=400

answer = None
def check_Timer_User_Input():
    time.sleep(20)
    if answer != None:
        return
    print ("WE ARE WAITING FOR YOUR CHOICE")
Thread(target = check_Timer_User_Input).start()

select_your_produc =input("select a product - COCA:260huf, FANTA:240, HOT_CHOCOLATE:350HUF, LATTE: 400HUF :- ")

def calculation(insert,product):
    
    if insert>=product:
        
        remain=insert-product
        if remain==0:
            print ("nothing to return")
        else:
            for i in Validated_coins:
                
                if i>remain:
                    pass
                else:
                    r=remain//i
                    remain=remain%i
                    t="return"+ " "+ str(r)+" "+str(i) +"huf"
                    l.append(t)
    else:
        print ("Entered price is not sufficient")
    print (" , ".join(l))

def vending_machine(n):
    n=n.upper()
    if n=="COCA":
        print ("plese enter huf:",COCA)
        product=COCA
        insert=int(input("enter cash"))
        if insert==5:
            print ("we dont accept 5 huf coins")
            insert=int(input("enter cash"))
        else:

            calculation(insert,product)
       
    elif n=="FANTA":
        print ("plese enter huf:",FANTA)
        product=FANTA
        insert=int(input("enter cash"))
        if insert==5:
            print ("we dont accept 5 huf coins")
            insert=int(input("enter cash"))
        else:

            calculation(insert,product)
       
    elif n=="HOT_CHOCOLATE":
        print ("plese enter huf:",HOT_CHOCOLATE)
        product=HOT_CHOCOLATE
        insert=int(input("enter cash"))
        if insert==5:
            print ("we dont accept 5 huf coins")
            insert=int(input("enter cash"))
        else:

            calculation(insert,product)
    elif n=="LATTE":
        print ("plese enter huf:",LATTE)
        product=LATTE
        insert=int(input("enter cash"))
        if insert==5:
            print ("we dont accept 5 huf coins")
            insert=int(input("enter cash"))
        else:

            calculation(insert,product)
    else:
        print ("sorry the product is not available")
    return "Thank you for your purchase"

vending_machine(select_your_produc)

        