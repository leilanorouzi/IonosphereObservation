import numpy as np
import pandas as pd
# import cmath
import reading_files
from parameters import *
from antenna import *
import matplotlib.pyplot as plt
from source import *
import magnetic_field
import geometry
from read_ionosphere import read_ionosphere
# import copy


class Medium(object):

    def __init__(self,filename_iono, filename_mag, column_names, num):
        
        self.fn_iono = filename_iono
        self.iono_df = read_ionosphere(filename_iono)

        # Column names
        self.source_names = column_names

        #number of sources
        self.source_num = num

        # reading magnetic values
        self.b0 =  magnetic_field.reading_igrf (filename_mag).loc[0, ['xcomponent', 'ycomponent', 'zcomponent']]*1*10**-9
        print()
        
        

    def max_height_ne(self):
        '''
        Calculate the hight of maximum electron desity
        :return:
        '''
        df = self.iono_df
        h = df.loc[df.N_e.argmax (), 'Height']
        n = df.N_e.max()

        return h,n


    def plot_iono(self):
        #Read the hieight and elecron densities
        max_ne_h,max_ne = self.max_height_ne()

        fig = plt.figure (figsize=(10, 10))
        fig.suptitle ('Please close the picture, when you finished', fontsize=20, color='brown')


        ax = plt.axes()
        plt.plot(self.iono_df.N_e,self.iono_df.Height)
        plt.title('The profile of the electron density vs. altitude\n Year=2020., Month= 07, Day= 04\nLatitude= 18.35, Longitude= 293.24')
        plt.xlabel(r'$n_e$, electron density')
        plt.ylabel(r'$h$, height (km)')

        plt.hlines(max_ne_h,0,max_ne,colors='orange',linestyles=':' )
        ax.text(max_ne/4,max_ne_h+20,r'$h$= {0:.0f} km, $max(n_e)=${1:2.3e}'.format(max_ne_h,max_ne),color='orange' )
        # max_ne_h
        plt.show()
        plt.close(fig)

        pass

    def ApHa_function(self, x,y,t,n,mode):
        '''
        Calculates different parts of Appleton-Hartree as n^2 = 1- numerator/denominator.
        :param x:
        :param y:
        :param t:
        :param n:
        :param mode:
        :return:
        '''
        # make a list out of a single number in order to be able to do apply mathematical functions on list parameters
        # construct an array with n*1 size and values of 1
        p1 = np.ones (n)
        # construct an array with n*1 size and values of 0.5
        ph = np.ones (n) * 0.5

        # calculate y*sin(theta)
        # construct an array with n*1 size and values of y*sin(theta)
        pys = np.multiply (ph, np.square ([y * np.sin (t)] * n).reshape ((n,)))
        # Calculate (y*sin(theta))^2
        pys2 = np.square (pys)

        #Calculate y*cos(theta)
        # construct an array with n*1 size and values of (y*cos(theta))^2
        pyc = np.square ([y * np.cos (t)] * n).reshape ((n,))

        #------------ Numerator ------------------
        # 1-X
        x1 = np.subtract (p1, x.values)
        # X^2
        x2 = np.square (x1)
        # numerator = (1-X)*X
        numerator = np.multiply (x.values, x1)

        #------------- Denominator ----------------
        # 0.5* (y*sin(theta))^2
        d0 = np.multiply (ph, pys2)

        # (1-X)*0.5*(y*sin(theta))^2
        d1 = np.subtract (x1, d0)

        # (0.5*(y*sin(theta))^2)^2
        d20 = np.square (d0)

        # (1-X)^2 * (y*cos(theta))^2
        d21 = np.multiply (x2, pyc)

        # (0.5*(y*sin(theta))^2)^2 + (1-X)^2 * (y*cos(theta))^2
        d2 = np.add (d20, d21)

        #√ ( (0.5*(y*sin(theta))^2)^2 + (1-X)^2 * (y*cos(theta))^2 )
        d3 = np.sqrt(d2)


        if mode=='O':
            # Calculate denominator for ordinary model
            denominator = np.add(d1, d3)
            # Calculate denominator for extra ordinary model
        else: denominator = np.subtract(d1, d3)

        # print(numerator)
        # print('\n\n')
        # print(denominator)

        # Calculate the refractive index
        d = np.divide(numerator,denominator)
        res = np.subtract(p1,d)
        return res


    def AppeltonHartree(self,k,f):
        # https://en.wikipedia.org/wiki/Appleton%E2%80%93Hartree_equation
        # https: // www.ferzkopp.net / Personal / Thesis / node7.html
        # assumption: collisionless plasma nu<<omega
        # the '+' sign represents the "ordinary mode," and the '−' sign represents the "extraordinary mode."


        nu = 0 # electron colission frequency, cold plasma

        # Electron density
        n = self.iono_df.N_e

        #--------------- Calculate the parameters of Appelton-Hartree -----------------------

        # The angular frequency w= 2pi*f. It will give a data frame.
        omega = pd.DataFrame(data=2 * np.pi * f). \
            rename(index={i: self.source_names[i] for i in f.index}) \
            .T \
            .rename(index={'f':0})

        #Electron plasma frequency. An array with n data points
        #w_0 = √(N*q_e^2/eps0*m_e)
        omega0 = np.sqrt ((n *  (q_e ** 2)) / (eps0 * m_e))

        # Electron gyro frequency
        # calculate the magnitude of ambient magnetic field
        # calculate w_h= b*q_e/m_e
        omegah = geometry.vec_mag(self.b0).real *  q_e / m_e

        # print(np.size(omegah),type(omegah))
        # print(omegah)

        # The angle between magnetic field and the wave number
        theta_b_k = pd.DataFrame(np.zeros((1,self.source_num)),columns=self.source_names)

        # Construct a dataframe for X
        x = pd.DataFrame(np.zeros((len(n),self.source_num)),columns=self.source_names)
        # Y = w_h /w
        y = omegah / omega

        # n^2 of Ordinary mode
        n_square_o = pd.DataFrame(np.zeros((len(n),self.source_num)),columns=self.source_names)
        # n^2 of extraordinary mode
        n_square_x = pd.DataFrame(np.zeros((len(n),self.source_num)),columns=self.source_names)

        # for every source calculate X, N^2 of ordinary and extraordinary mode
        for i,s in enumerate(self.source_names):
            # calculate the angle between magnetic field anf the wave number of the source
            theta_b_k.loc[:,s] = geometry.vectors_angel(k.loc[i],self.b0)
            # print (s, geometry.vectors_angel (k.loc[i], self.b0).real,theta_b_k[s])

            # X = (w_0 / w)^2
            x[s] = np.square (omega0 / omega[s].to_list()*len(n))


            #-------------------Refractive index ---------------------------
            # Calculate square value of ordinary and extraordinary modes of  refractive index

            # ordinary mode
            # n_square_o = 1 - (
            #         (x * (1 - x)) /
            #         (1 - x - (0.5 * np.square (y * np.sin (theta_b_k))) + \
            #          np.sqrt (np.square (0.5 * np.square (y * np.sin (theta_b_k))) + \
            #                   np.square ((1 - x) * y * np.cos (theta_b_k))))
            # )
            n_square_o[s] = self.ApHa_function(x=x[s],y=y[s],t=theta_b_k[s],n=len(n),mode = 'O')

            # # extraordinary mode
            # n_square_x = 1 - (
            #         (x * (1 - x)) /
            #         (1 - x - (0.5 * np.square (y * np.sin (theta_b_k))) - \
            #          np.sqrt (np.square (0.5 * np.square (y * np.sin (theta_b_k))) + \
            #                   np.square ((1 - x) * y * np.cos (theta_b_k))))
            # )
            n_square_x[s] = self.ApHa_function (x=x[s], y=y[s], t=theta_b_k[s], n=len (n), mode='X')


        print()




        return n_square_o,n_square_x


# filename_iono = 'Data/Input/ionospheric_parameters.txt'
# filename_mag = 'Data/Input/igrfwmmData.json'
# sfn = '../IonosphereObservation/Data/Input/source2.txt'
#
# source = Source(source_fn=sfn, dipole= False)
#
# iono = Medium(filename_iono= filename_iono, filename_mag = filename_mag, column_names=source.s_columns, num=source.source_n)
#
# # iono.plot_iono()
# print(iono.AppeltonHartree(k=source.source_charc.k , f=source.source_charc.f*10**6))