# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 22:15:02 2018

@author: Antika
"""
import sys
  

def iterate_prv_nxt(Query_W):
    prv, cur, nxt = None, iter(Query_W), iter(Query_W)
    next(nxt, None)

    while True:
        try:
            if prv:
                yield next(prv), next(cur), next(nxt, None)
            else:
                yield None, next(cur), next(nxt, None)
                prv = iter(Query_W)
        except StopIteration:
            break
Query_W=[]
Ans_A=[]
l=[]
Query = str(input('enter the query(W) '))
Ans =   str(input('enter the ans(A): '))
for i in range(len(Query)):
    Query_W.append(Query[i])
    
for i in range(len(Ans)):
    Ans_A.append(Ans[i]) 

if (len(Query_W)==len(Ans_A)):
   
    
    for prv, cur, nxt in iterate_prv_nxt(Query_W):
        pass
    
#        print (prv, cur, nxt)
   
        for j in range(len(Ans_A)):
        
        
            if (Ans_A[j]==prv or Ans_A[j]==cur or Ans_A[j]==nxt):
                pass
            
        l.append(Ans_A[j])
            
print("\n")
if (len(l)==(len(Ans_A))):
    print ("the ans is acceptable")
else:
    print ("the ans is NOT acceptable ")
        

    
        

        
        
          