import warnings

warnings.filterwarnings("default", category=DeprecationWarning)

import numpy as np
# from multi_source import *
import wave_propagation as wp
from visualization import *
# from source_antenna import *
# from antenna import *
import os
import sys
from tkinter import Tk
from tkinter import filedialog


def select_directory():
    '''
    This function, opens a window and asks the user to choose the folder that contains the files

    :return: the path of the input data
    :rtype: str
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
    # -------------------------------------------------
    # Getting input data. There are 2 ways to import the data
    # 1- You can select the data folder
    # input_dir = select_directory()
    # 2- or you can type it in here directly.
    # If you would like to use this method please, comment above line and uncomment following line.
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    input_dir = "../IonosphereObservation/Data/Input/"



    # file name of the antenna
    rfn = os.path.join(input_dir,'03-tri50_2.txt')
    # file name of the source
    sfn = os.path.join(input_dir, 'source2.txt')

    # -------------------------------------------------
    # makeing an instance of the class
    w_p = wp.WavePropagation(source_fn=sfn, radar_fn=rfn, dipole= True)

    #-------------------------------------------------
    # Visualization
    # Plotting the antennas and source/s location

    # Calling the visualization calss
    vis = visualization(a=w_p.antenna_location, s=w_p.source_location)
    # To obtain the location of the antenna and source and plot them
    print("\x1b[1;31m Please check the plot window.\x1b[0m\n")
    vis.source_antenna_location()

    # -------------------------------------------------
    # Calculations

    # To obtain waves at the antenna
    # run antenna_wave_received function to calculate the wave results as a data frame for all sources and antennas
    w = w_p.antennna_wave_received()
    # To obtain the total result for each antenna, call the vector_superposition function.
    # It calculate received waves from all sources for each antenna in the original reference frame
    # and add their components up.
    waves = w_p.vector_superposition(w)
    # print('\x1b[1;31mReceived waves from each source at the antenna locations:\x1b[0m \n', waves)

    # To obtain the phase difference at the antenna call phase_diff function.
    phase_difference = w_p.phase_diff()
    print('\nPhase difference in degree:\n', np.degrees(phase_difference).round(3))
    print("\x1b[1;31m===============================================================================\n\n\x1b[0m")
    #
    # # To obtain the voltage call voltage function
    # voltage = ms.voltage()
    #
    # -------------------------------------------------
    # The end
    print('FINISHED')

    pass


if __name__ == '__main__':
    main()
