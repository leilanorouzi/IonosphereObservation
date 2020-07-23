import numpy as np
import pandas as pd
import cmath
import reading_files 
from parameters import *
from antenna import *


class Source(object):
    '''
    :param source_fn: Source/s txt file path
    :type source_fn: str
    :param dipole: If the source is a dipole transmitter or not. The default value is false.
    :type dipole: Boolean
    '''
    def __init__(self,source_fn,dipole):

        self.source_fn = source_fn

        self.dipole = dipole
        self.source_charc = self.read_source()
        self.source_n = len(self.source_charc)
        self.s_columns = ['s'+str(i+1) for i in range(self.source_n)]

        # In free space it is equal to 1 otherwise it is a matrix
        self.ref_index = np.sqrt(1 - (80.616 * n_e / self.source_charc.f ** 2))

        # Calculating the wave number and adding that to the source_charc dataframe
        self.source_charc = self.wave_num()

        # Taking coordination and to convert to meter
        self.source_location =  self. source_charc.loc[:,['x','y','z']].values * 1000

        # -------------- polarization parameters ----------------
        self.polar_stat = 'Unpolarized'
        self.alpha_x = 0
        self.alpha_y = 0

        # -------------- Dipole transmitter parameters-------------
        # The current of the transmitter
        self.I0 = 0.001  # mAm
        # finding the angle for every set of antenna and the source.
        # these are the angles of the dipole source from z and x directions.
        # you can change them
        self.theta_z = np.pi / 3  # The angle between dipole and  z axis
        self.theta_x = np.pi / 8  # The angle between dipole and x axis

    def read_source(self):
        '''

        :return:
        '''

        # the name of the columns
        column_names = ['x','y','z','A_s','theta_s','f']

        # To open the file
        read = open(self.source_fn, 'r')
        i = 0  # The number of the row of the file

        # An empty data farme to be filled with the source values regarding to the column names
        source_df = pd.DataFrame(columns=column_names)

        for line in read:
            newline = line.rstrip("\n")
            if (newline!='' and newline[0] not in ['#','%']):
                newline = reading_files.read_line(line,column_names)
                source_df = source_df.append(newline,ignore_index=True)
        read.close()
        print('\x1b[1;33mSource variables values:\x1b[0m \n', source_df)
        return source_df

    def wave_num(self):
        '''

        :return:
        '''

        # Wavenumber k= n*w/c
        # n:refractive index ,
        # w: the angular frequency w=2*pi*f
        # k= k_x i+ k_y j+ k_z k
        # in here (k_x^2+k_y^2+k_z^2)^0.5= self.k
        # for each source we have self.k=[k_x,k_y,k_z] , since traveling along z direction
        # kx=ky=0
        # kz=ks
        df = self.source_charc
        ks = round(2 * np.pi * self.ref_index * (df.f * (10 ** 6)) / c0, 4)

        df['k'] = None
        for i in range(self.source_n):
            df.loc[[i], 'k'] = pd.Series([[0, 0, ks[i]]], index=[i])
        return df

    def polarization(self) -> np.array:
        '''
        This function calculate the Jones vector of a situation. As, e^(i(a_x,a_y,0)).
        The parameters are set at the parameter.py file.

        :param polar_stat: The type of polarization. unpolarized, linear, elliptical
        :type polar_stat: str
        :param alpha_x: The phase angle respect to x direction
        :type alpha_x: float
        :param alpha_y: The phase angle respect  to y direction
        :type alpha_y: float
        :return: The jones vector
        :rtype: numpy.array
        '''

        polar_stat = self.polar_stat
        alpha_x = self.alpha_x  # phase angle
        alpha_y = self.alpha_y  # pahse angle

        a = np.zeros([self.source_n,3],dtype=complex)

        if polar_stat.lower()=='unpolarized':
            if ((alpha_x!=0) & (alpha_y!=0)) :
                print("\x1b[1;31m THE PARAMETER DON'T FIT THE POLARIZATION STATUS, please check again.\x1b[0m\n" )
            else:
                print("\x1b[1;31m THERE IS NO POLARIZATION.\x1b[0m\n")
        else:

            if polar_stat.lower()=='linear':
                #Ex and Ey have the same phase (but different magnitude)
                #https://en.wikipedia.org/wiki/Linear_polarization
                if ((alpha_x == alpha_y) & (alpha_y!=0)):
                    print("\x1b[1;31m THE PARAMETER DON'T FIT THE POLARIZATION STATUS, please check again.\x1b[0m\n")
                else:
                    print("\x1b[1;31m THE POLARIZATION IS LINEAR.\x1b[0m\n")

            if polar_stat.lower()=='elliptical':
                if alpha_x == alpha_y:
                    print("\x1b[1;31m THE PARAMETER DON'T FIT THE POLARIZATION STATUS, please check again.\x1b[0m\n")
                else:
                    print("\x1b[1;31m THE POLARIZATION IS ELLIPTICAL.\x1b[0m\n")

        for s in range(self.source_n):
            a[s,:] = [ cmath.exp(complex(0, 1)*alpha_x) ,
                       cmath.exp(complex(0, 1)*alpha_y),
                       0]
        return a


