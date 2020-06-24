'''


'''

from reading_files import *
from parameters import *
from reference_transformation import *
import cmath


class multi_source(object):
    '''
    :argument:
        - radar_fn: the file name containing the values of the antenna variables
        - source_fn: the file name containing the values of the wave source variables
    :attributes:
        - antenna_charc: DataFrame, Antenna variables
        - antenna_location: Array, antenna location coordinate
        - radar_n: Integer: The number of antenna

        - ref_index: refractive index of the medium

        - source_charc: Dataframe, source of wave variables
        - source_n: Integer:, the number of the wave sources
        - s_columns: List, the header of the column name for creating the DataFrame
        - source_location: Array, the location of wave sources

        - distance: DataFrame, the distance between source/s and antennas
        - path_vec: DataFrame, the vecors of path from anttena to source/s

    :methods:
        - dist: to calculate the distance between two points in Cartesian coordinate
        - multi_dist: to caculate the distance and the vector between each set of source-antenna. It calls dist function to calculate the distance.
        - polarization: To calculate the Jones vector of the source. It gets phase angels.
        - antennna_wave_received: This function calculate the field components and the phase of the field received at the antenna location, trasmitted from the source. The results are calculated at the ray path attached reference frame.
        - vector_superposition: Calculate the superposition of the vecor by rotating the refernce frame from ray path attached reference frame to the original reference frame.
        - voltage: Calculate the volatge of the received electric field at the antenna location.

    :Example:
    >>> from multi_source import *
    >>> ms = multi_source(radar_fn,source_fn)
    >>> phase, wave = ms.antennna_wave_received()
    '''

    def __init__(self,radar_fn:str,source_fn:str):
        '''

        :param radar_fn: Antenna txt file path
        :type radar_fn: str
        :param source_fn: Source/s txt file path
        :type source_fn: str

        '''

        self.antenna_charc = read_antenna(radar_fn)
        self.antenna_location = self.antenna_charc.loc[:, ['x', 'y', 'z']].values  # Taking coordinations
        self.radar_n = len(self.antenna_charc)

        self.ref_index = n # In free space it is equal to 1 otherwise it is a matrix

        self.source_charc = read_source(source_fn)
        self.source_n = len(self.source_charc)
        self.s_columns = ['s'+str(i) for i in range(self.source_n)]

        # The wavelength of the source.
        # self.lambda_s = c0 / (self.source_charc.f.values * (10 ** 6))

        # Wavenumber k= n*w/c
        # n:refractive index ,
        # w: the angular frequency w=2*pi*f
        # k= k_x i+ k_y j+ k_z k
        # in here (k_x^2+k_y^2+k_z^2)^0.5= self.k
        # for each source we have self.k=[k_x,k_y,k_z] , since traveling along z direction
        #kx=ky=0
        # kz=ks
        ks=round(2*np.pi*self.ref_index*(self.source_charc.f * (10 ** 6))/c0,4)

        self.source_charc['k'] = None
        for i in range(self.source_n):
            self.source_charc.loc[[i], 'k'] = pd.Series([[0,0,ks[i]]], index=[i])
        self.source_location =  self. source_charc.loc[:,['x','y','z']].values * 1000    #Taking coordination and to convert to meter

        # Source to antenna distances
        self.distance, self.path_vec = self.multi_dist()
        print("Distance:\n", self.distance,
              "Vector:\n", self.path_vec)

    def dist(self,a:list,b:list)->float:
        '''
        This function calculates the distance between two points

        :param a: first location coordinates
        :type a: list
        :param b: second location coordinates
        :type b: list
        :return: the distance
        :rtype: float
        '''
        return np.sqrt( (a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2 )


    def multi_dist(self):
        '''
        This function calculates the distance between source/s and antenna/s and also returns a vector of source-antenna for every set of source-antenna.

        :returns:
            - dist_arr: the distances from the source to the antenna
            - sa_vec:  the vector from source to the antenna in Cartesian coordinate system (x,y,z).
                        each elements are a list of vector components
        :rtype: pandas.Dataframe


        '''

        dist_arr = pd.DataFrame(np.zeros((self.radar_n,self.source_n)),
                              columns=self.s_columns)
        sa_vec = pd.DataFrame(np.zeros((self.radar_n,self.source_n)),
                              columns=self.s_columns)

        for i in range(self.radar_n):
            for j in range(self.source_n):
                dist_arr.iloc[i,j] = self.dist(self.source_location[j,:], self.antenna_location[i,:])
                sa_vec.iloc[[i],j] = pd.Series([-self.source_location[j,:]+ self.antenna_location[i,:]],index=[i])
        return dist_arr,sa_vec

    def polarization(self,
                     polar_stat='unpolarized',
                     alpha_x=0, #phase angle
                     alpha_y=0, #pahse angle
                     ) -> np.array:
        '''
        This function calculate the Jones vector of a situation. As, e^(i(a_x,a_y,0)).

        :param polar_stat: The type of polarization. unpolarized, linear, elliptical
        :type polar_stat: str
        :param alpha_x: The phase angle respect to x direction
        :type alpha_x: float
        :param alpha_y: The phase angle respect  to y direction
        :type alpha_y: float
        :return: The jones vector
        :rtype: numpy.array
        '''
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


    def antennna_wave_received(self) ->( pd.DataFrame , pd.DataFrame ):
        '''
        This function calculates the value of the recieved wave from every source at all antenna.
        The assumptions:
            - Plane wave
            - Free space
            - The wave propagates along z direction

        The wave considered as a combination of amplitude, Amp and oscillation, osc.
        Oscillation part contains the phase.

        :return:

            -phase: Dataframe, the phase of wave from each source at the antenna location

            -wave: Dataframe, the received wave from each source at the location of antenna
        '''



        # traveling time
        t = self.distance / (c0/self.ref_index)

        # Amplitudes
        #     Amp[:,i] = [Amp0[i],0,0]  # plane wave traveling along z direction
        # assuming Ey=Ez=0
        Amp = np.zeros([self.source_n,3])
        # For every source gets the amplitude
        for s in range(self.source_n):
            Amp[:,s] = self.source_charc.A_s[s]


        # Phase:  w*t-k.r+phi
        #a=k.r
        w_num = self.source_charc.k

        # For every antenna and source calculates k.r
        a = np.array([[
            np.dot(w_num[i], [0,0,self.distance.iloc[j,i]])
            for j in range(self.radar_n)]
            for i in range(self.source_n)]).T
        # Convert the array to a DataFrame
        a = pd.DataFrame(a,columns=self.s_columns)

        # b=w*t
        # For every source calculate temporal part of oscillation w.t
        b = 2 * np.pi * (10 ** 6) * \
            np.array([self.source_charc.loc[i, 'f'] * t.iloc[:, i] for i in range(self.source_n)]).T
        # Convert it to a DataFrame
        b = pd.DataFrame(b,columns=self.s_columns)

        # Calculate the phase as k.r - w.t
        phase = a-b

        # Calculate the oscillation as: exp i(k.r-wt)
        osc = phase.applymap(lambda x: cmath.exp(complex(0, 1)*x))

        # polarization part
        # Obtain the jones vector by running polarization function
        Jones_vec = self.polarization(polar_stat='Unpolarized',
                                      alpha_x=0,
                                      alpha_y=0)

        print(
              '\nTime:\n', t,
              '\nAmplitude:\n',Amp,
              '\nPhase:\n',phase,
              '\nOscillation:\n',osc,
              '\nPolarization:\n',Jones_vec)

        # Make an empty DataFrame for result of the received field
        wave = pd.DataFrame(np.zeros([self.radar_n,self.source_n]),columns=self.s_columns)
        # Calculate the wave equation for every set of antenna-source
        # wave= amp*real(Jones_vec*osc) , plane wave
        for j in range(self.source_n):
            for i in range(self.radar_n):
                non_amp = osc.iloc[i, j]*Jones_vec[j,:]
                w = [Amp[j,:]* non_amp.real]
                wave.iloc[[i],j] = pd.Series(w,index=[i])

        print('\x1b[1;31mReceived waves from each source at the antenna locations:\x1b[0m \n',wave)
        return phase,wave
        # pass

    def vector_superposition(self,w)->list:
        '''
        Gets the received waves from every sources at the antenna.
        Rotate ray path reference frame to the original reference frame.
        add them up to calculate the total waves from all sources.

        :param w: The result of wave from antennna_wave_received function.
        :type w: pd.DataFrame
        :return: The superposition of waves received from source/s for each antenna
        :rtype: list
        '''

        # Build an empty array for final result
        total_waves = np.zeros([self.radar_n,3],dtype=complex)

        for i_a in range(self.radar_n):  #for every antenna

            antenna_w_total = w.iloc[i_a,0]

            for i_s in range(self.source_n):
                if i_s !=0:
                    # Calculate the rotation matrix to rotate ray path attached reference frame to the original refernce
                    rotation_matrix = rotate_refernce(self.path_vec.iloc[i_a,i_s])
                    # Obtain the field vector in the original reference frame by multiplying rotation matrix to the field vector
                    rotated = np.dot(rotation_matrix,w.iloc[i_a,i_s])

                    # Adding the field vectors from all sources
                    antenna_w_total = np.add(
                        rotated,
                        antenna_w_total)
            total_waves[i_a] = antenna_w_total

        print('\x1b[1;31mTotal Wave form at the antenna locations:\x1b[0m \n',total_waves)

        return total_waves

    def phase_diff(self) ->pd.Series:
        '''
        Calculates the phase difference at antenna. Takes the phase part from antennna_wave_received function and for each antenna add them up.

        :return:
            - The phase difference of received waves at every antenna
        '''
        #phi =atan2 (Im(z),Re(z))
        #z = r e^(i*phi)
        p_, w_ = self.antennna_wave_received()
        print(p_)

        # Add all phases from all sources to find the total phase at every antenna
        phase_diff = p_.sum(axis=1)

        print('\x1b[1;31mPhase differences at the antenna locations (rad):\x1b[0m \n')
        for i in phase_diff: print(i)
        return phase_diff


    def voltage(self)-> np.array:
        '''
        multiplies the total electric field to the length of the of antenna in each direction
        :return: The voltage at the antenna
        '''
        total_voltage=0
        l = 3.8     #antenna length
        antenna_length = [l,l,0]
        p,w_ = self.antennna_wave_received()

        w = self.vector_superposition(w_)

        total_voltage = list(map(lambda x : x.real*antenna_length,w))
        # total_voltage = total_voltage.applymap(lambda x: np.sqrt(np.dot(x,x)))

        print('\x1b[1;34mVoltages at antenna:\n\x1b[0m', total_voltage)
        return total_voltage


    # def attenuation(self):
    #     # No scattering or absorption
    #     # From inverse square law
    #     # calculate the attenuation of the wave at the antenna location
    #     #Taking the real part of the waveform at the antenna location
    #     dist, _ = self.multi_dist()
    #     _ , w = self.antennna_wave_received()
    #     poynting_vector = w.applymap(lambda x: np.dot(x,np.conjugate(x)*0.5/eta).real)
    #     # print('\x1b[1;34mWave_vec : \n\x1b[0m',poynting_vector)
    #
    #     # poynting_vector =np.divide(poynting_vector.values,np.square(dist).values)
    #
    #     print('\x1b[1;34mThe intensities from the source/s are:\n\x1b[0m', poynting_vector, "")
    #     return poynting_vector



pass
# #
# fnr = '../IonosphereObservation/Data/Input/03-tri50_2.txt'
# fns = '../IonosphereObservation/Data/Input/source2.txt'
# #
# ms = multi_source(radar_fn=fnr,source_fn=fns)
# ms.antennna_wave_received()
# ms.phase_diff()

