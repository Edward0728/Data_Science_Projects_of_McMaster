# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 20:29:34 2020

@author: Seshasai
"""

import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import numpy as np
import time
import sys

class Room:
    def __init__(self, length, width):
    # intialize two line objects (one in each axes)
        self.fig1, (self.ax1, self.ax2)=plt.subplots(2,1)
        self.line1, = self.ax1.plot([], [], lw=2)
        self.line2, = self.ax2.plot([], [], lw=2, color='r')
        self.line = [self.line1, self.line2]
        self.__length = length
        self.__width = width
    
    def get_area(self):
        self.__area = self.__length*self.__width
        return self.__area

    def updateGrid(a, b):
       count = 0
       particularBox_xCoord = [(b-1)*0.1, (b-1)*0.1,(b)*0.1, (b)*0.1,(b-1)*0.1 ]
       particularBox_yCoord = [(a-1)*0.1, (a)*0.1,(a)*0.1, (a-1)*0.1, (a-1)*0.1]
       plt.plot(particularBox_xCoord, particularBox_yCoord,'blue')
       #line2.set_data(particularBox_xCoord,particularBox_yCoord)
       #plt.show()
       count = count + 1
    
    def calWork(self):
       option = 0
       bots = int(input('''Every robot clean 0.1 sqaure units per minute. 
       Please enter the number of robots in use(1, 10): \n'''))
       while option != 1 and option != 2 and option != 3:
           print('')
           print('********OPTION MENU********')
           print('Option 1: How much percentage of the room do you want to clean? ')
           print('Option 2: How long do you you want to clean in minutes')
           print('Option 3: Exit')
           option = int(input('Please choose a task: '))
       
       if option == 1:
           percentage = float(input("Please enter the percentage in decimal: "))
           print('Estimated work time: ', self.__area*percentage/(bots*0.1), ' minutes.')
       if option == 2:
           time = int(input("Please enter the time in minutes: "))
           print('Estimated clean area percentage: ', 100*bots*0.1*time/self.__area, '%')
       if option == 3:
           sys.exit()
            
    done_list = []
    y1data = []
    def generatorFunc(self,i,length, width):
       for j in range (int(length*width*300)):
       
           #a = random.randint(1,int(10)) # row number
           #b = random.randint(1,int(10))# column number
           a = random.randint(1,int(length*10)) # row number
           b = random.randint(1,int(width*10))# column number
           print(a, b)
       
           self.ax1.set_ylim(0, 100)
           self.ax1.set_xlim(0, int(length*width*300))
           if [a, b] not in Room.done_list:          
               Room.done_list.append([a, b])
               percent =100*len(Room.done_list)/(length*width*100)
               time.sleep(1)
               print('Completed ', percent, '%.')
               self.y1data.append(percent)
       
           if len(Room.done_list) == length*width*100:
               print('Done!')
               #time.sleep(5)
           #return 100*len(done_list)/(length*width*100)
       
           #y1data.append(percent)
       
           self.ax1.set_ylim(0, 100)
           self.ax1.set_xlim(0, 100)
           self.line[0].set_data(len(Room.done_list),  self.y1data)
       
           self.x1data = list(range(0,len( Room.done_list)))
           self.ax1.plot( self.x1data,  self.y1data)
       
           self.ax2.set_ylim(-1.1, 1.1)
           self.ax2.set_xlim(0, width)
           self.ax2.grid()
       
           Room.updateGrid(a, b)
           
           return  self.line

def main():    
    length = float(input('Please enter the length of the room (0.3, 5): '))
    width = float(input('Please enter the width of the room (0.3, 1): '))  
    My_instance = Room(length, width)
    #A.set_area()
    area = My_instance.get_area()
    My_instance.calWork()
    #done_list = generatorFunc(length, width)
    # box number to colour

    ani = animation.FuncAnimation( My_instance.fig1, My_instance.generatorFunc, fargs = (length, width), interval=3000, repeat = False)
  
    plt.show()
    #sys.exit(0)
 
main()    
        
    