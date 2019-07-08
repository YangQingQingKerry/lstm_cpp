#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random as rd


class Battleship:
    def __init__(self, ships, grid):
        self.ships=ships
        self.grid=grid
        self.n=len(grid)


    def cond1(self, pt): #position occupied?
        if self.grid[pt[0]][pt[1]]==1:
            return False
        else:
            return True


    def cond2(self, pt, ship): #left-right-up-down suitable?
        i=pt[0]
        j=pt[1]
        self.states=[False]*4
        #left
        if j>=ship-1:
            if sum([self.grid[i][j-k] for k in range(ship)])==0:
                self.states[0]=True
        #right
        if (self.n-j)>=ship:
            if sum([self.grid[i][j+k] for k in range(ship)])==0:
                self.states[1]=True
        #up
        if i>=ship-1:
            if sum([self.grid[i-k][j] for k in range(ship)])==0:
                self.states[2]=True
        #down
        if (self.n-i)>=ship:
            if sum([self.grid[i+k][j] for k in range(ship)])==0:
                self.states[3]=True

        if len(self.states)==0:
            return False
        else:
            return True


    def lineup(self):
        for ship in self.ships:
            i, j=rd.randint(0, self.n-1), rd.randint(0, self.n-1)
            while not (self.cond1([i,j]) & self.cond2([i,j], ship)):
                i, j=rd.randint(0, self.n-1), rd.randint(0, self.n-1)

            '''decision'''
            Ava=[i for i in range(4) if self.states[i]==True]
            sd=rd.choice(Ava)
            if sd==0:
                for k in range(ship):
                    self.grid[i][j-k]=1
            elif sd==1:
                for k in range(ship):
                    self.grid[i][j+k]=1
            elif sd==2:
                for k in range(ship):
                    self.grid[i-k][j]=1
            else:
                for k in range(ship):
                    self.grid[i+k][j]=1


    def Single(self):
        self.rid=np.array(self.grid)
        Avaset=[(i,j) for i in range(self.n) for j in range(self.n)]



        while sum(sum(self.rid))>0:
            i, j=rd.choice(Avaset)
            Avaset.remove((i,j))

            if self.rid[i][j]==1:
                self.rid[i][j]=0
                tg=[(i,j)]
                _next=[(i+k,j) for k in [-1,1] if (i+k,j) in Avaset]+[(i, j+k) for k in [-1,1] if (i,j+k) in Avaset]
                while len(_next)>0:
                    message=''
                    for pt in _next:
                        Avaset.remove(pt)
                        if self.rid[pt[0]][pt[1]]==1:
                            message='hit'
                            self.rid[pt[0]][pt[1]]=0
                            tg.append(pt)
                            break
                    if message=='hit':
                        if tg[0][0]==tg[-1][0]:
                            j_min=min([item[1] for item in tg])
                            j_max=max([item[1] for item in tg])
                            _next= {(tg[0][0], j_min-1), (tg[0][0], j_max+1)} & set(Avaset)
                        else:
                            i_min=min([item[0] for item in tg])
                            i_max=max([item[0] for item in tg])
                            _next= {(i_min-1, tg[0][1]), (i_max+1, tg[0][1])} & set(Avaset)
                    else:
                        break
        self.steps=self.n**2-len(Avaset)






if __name__=="__main__":

    grid=[[0 for i in range(10)] for j in range(10)]
    ships=[5, 4, 3,3,2]

    ls=Battleship(ships, grid)
    ls.lineup()
    for line in ls.grid:
        print(line)

    print("\n\n")
    battle=ls.Single()
    for line in ls.rid:
        print(line)
    print(ls.steps)
