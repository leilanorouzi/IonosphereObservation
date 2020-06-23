import warnings
warnings.filterwarnings("default", category=DeprecationWarning)

import numpy as np
from multi_source import *
from visualization import *
import os
# import tkinter as tk
from tkinter import Tk
from tkinter import filedialog

def select_directory():
    '''
    This function, opens a window and asks the user to choose the folder that contains the files

    :return: the path of the input data
    :rtype:str
    '''

    # You can choose the directory of the input data
    root = Tk()
    root.geometry('200x150')
    root.withdraw()
    input_dir = filedialog.askdirectory(parent=root,
                                        title='Choose the directory of the data',
                                        initialdir=os.getcwd(),
                                        mustexist=True) + '/'

    # root.mainloop()
    root.destroy()

    return input_dir

def main():
    '''
    The main code to run other modules and plot the graphics.

    :return: None
    '''

    #You can select the data folder
    input_dir = select_directory()
    # or you can type it in here directly
    # input_dir = "../IonosphereObservation/Data/Input/"

    # file name of the antenna
    fnr = input_dir+'03-tri50_2.txt'
    # file name of the source
    fns = input_dir+'source2.txt'

    #to call the class
    ms = multi_source(radar_fn=fnr,source_fn=fns)

    #Visulization part
    vis = visualization(a=ms.antenna_location, s=ms.source_location)
    # To obtain the location of the antenna and source and plot them
    print("\x1b[1;31m Please check the plot window.\x1b[0m\n")
    vis.source_antenna_location()

    # To obtain waves at the antenna
    _,w = ms.antennna_wave_received()
    waves = ms.vector_superposition(w)


    # To obtain the phase difference at the antenna
    phase_difference =  ms.phase_diff()
    print('\nPhase difference in degree:\n',np.degrees(phase_difference).round(3))

    voltage = ms.voltage()


    print('FINISHED')

    pass

if __name__ == '__main__':
    main()


