import pandas as pd
# from reading_files import *
from parameters import *
import geometry
from source_antenna import SourceAntenna
import medium
import cmath

class WavePropagation(SourceAntenna):
    '''
    :argument:
        - radar_fn: the file name containing the values of the antenna variables
        - source_fn: the file name containing the values of the wave source variables
        - dipole: If the source is a dipole transmitter or not?
    :attributes:

        - ref_index: refractive index of the medium

    :methods:
        - phase : To calculate the phase
        - antennna_wave_received: This function calculate the field components and the phase of the field received at the antenna location, trasmitted from the source. The results are calculated at the ray path attached reference frame.
        - vector_superposition: Calculate the superposition of the vecor by rotating the refernce frame from ray path attached reference frame to the original reference frame.
        - voltage: Calculate the volatge of the received electric field at the antenna location.

    :Example:

    >>> from wave_propagation import *
    >>> wp = WavePropagation(radar_fn,source_fn,dipole=True)
    >>> wave = wp.antennna_wave_received()
    '''

    def __init__(self,radar_fn:str,source_fn:str, dipole=False):
        '''
        :param radar_fn: Antenna txt file path
        :type radar_fn: str
        :param source_fn: Source/s txt file path
        :type source_fn: str
        :param dipole: If the source is a dipole transmitter or not. The default value is false.
        :type dipole: Boolean
        '''

        # Feeding inputs to the SourceAntenna calls to make an object of the source and the antenna
        SourceAntenna.__init__(self,source_fn, radar_fn, dipole)

        filename_iono = 'Data/Input/ionospheric_parameters.txt'
        filename_mag = 'Data/Input/igrfwmmData.json'


        # Feeding inputs to the medium class to get medium properties
        medium.Medium.__init__(self, filename_iono= filename_iono, filename_mag = filename_mag, column_names=self.s_columns, num=self.source_n)

        # In free space it is equal to 1 otherwise it is a matrix

        self.ref_index = np.sqrt(1 - (80.616*n_e / self.source_charc.f**2 ))


        # traveling time
        self.time = np.divide(self.distance , c0/self.ref_index)

        print('\n\x1b[1;31mTime:\n\x1b[0m', self.time,
            '\n\x1b[1;31mAmplitude:\n\x1b[0m', self.source_charc.A_s)


    def wtkr(self)->pd.DataFrame:
        '''
        To calculate the oscillation part of wave form.
        :return:
        '''

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
            np.array([self.source_charc.loc[i, 'f'] * self.time.iloc[:, i] for i in range(self.source_n)]).T
        # Convert it to a DataFrame
        b = pd.DataFrame(b,columns=self.s_columns)

        # Calculate the phase as k.r - w.t
        res = a-b
        return res



    def phase_diff(self) ->pd.Series:
        '''
        Calculates the phase difference at antenna. Takes the phase part from antennna_wave_received function and for each antenna add them up.

        :return:
            - The phase difference of received waves at every antenna
        '''

        if self.dipole :
            _,dipole_transmitter_pahse = self.dipole_transmitter()
        else:
            dipole_transmitter = 0
        #phi =atan2 (Im(z),Re(z))
        #z = r e^(i*phi)
        # the final phase = phase from k.r-wt term + phase from dipole transmitter
        p_= self.wtkr() + (np.arcsin(dipole_transmitter_pahse).divide(self.distance))
        print(p_)

        # Add all phases from all sources to find the total phase at every antenna
        phase_diff = p_.sum(axis=1)

        print('\x1b[1;31mPhase differences at the antenna locations (rad):\x1b[0m \n')
        for i in phase_diff: print(i)
        return phase_diff


    def antennna_wave_received(self) ->pd.DataFrame :
        '''
        This function calculates the value of the recieved wave from every source at all antenna.
        The assumptions:
            - Plane wave
            - Free space
            - The wave propagates along z direction

        The wave considered as a combination of amplitude, Amp and oscillation, osc.
        Oscillation part contains the phase.

        :return:
            -wave: Dataframe, the received wave from each source at the location of antenna
        '''
        # from PIL import Image
        # myImage = Image.open("sphinx/_static/dipole_3d_annotated.png");
        # myImage.show();

        # traveling time
        t = self.time

        # Amplitudes
        #     Amp[:,i] = [Amp0[i],0,0]  # plane wave traveling along z direction
        # assuming Ey=Ez=0
        Amp = np.zeros([self.source_n,3])
        # For every source gets the amplitude
        for s in range(self.source_n):
            Amp[:,s] = self.source_charc.A_s[s]

        #calculate the phase
        phase = self.wtkr()

        # Calculate the oscillation as: exp i(k.r-wt)
        osc = phase.applymap(lambda x: cmath.exp(complex(0, 1)*x))
        # consider a dipole source in the ionosphere
        if self.dipole:
            dipole_ , _= self.dipole_transmitter()
            osc_dipole = osc.multiply(dipole_)
        else: osc_dipole = osc

        # polarization part
        # Obtain the jones vector by running polarization function
        Jones_vec = self.polarization()

        # print(
        #       '\nPhase, no polarization, no dipole:\n',phase,
        #       '\nOscillation:\n',osc,
        #       '\nPolarization:\n',Jones_vec)

        # Make an empty DataFrame for result of the received field
        wave = pd.DataFrame(np.zeros([self.radar_n,self.source_n]),columns=self.s_columns)
        phase_final = pd.DataFrame(np.zeros([self.radar_n,self.source_n]),columns=self.s_columns)

        # Calculate the wave equation for every set of antenna-source
        # wave= amp*real(Jones_vec*osc) , plane wave
        for j in range(self.source_n):
            for i in range(self.radar_n):
                non_amp = osc_dipole.iloc[i, j]*Jones_vec[j,:]
                w = [Amp[j,:]* non_amp.real]
                wave.iloc[[i],j] = pd.Series(w,index=[i])

        return wave
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
                    rotation_matrix = geometry.rotate_refernce(self.path_vec.iloc[i_a,i_s])
                    # Obtain the field vector in the original reference frame by multiplying rotation matrix to the field vector
                    rotated = np.dot(rotation_matrix,w.iloc[i_a,i_s])

                    # Adding the field vectors from all sources
                    antenna_w_total = np.add(
                        rotated,
                        antenna_w_total)
            total_waves[i_a] = antenna_w_total

        print('\x1b[1;31mTotal Wave form at the antenna locations:\x1b[0m \n',total_waves)

        return total_waves



    def voltage(self)-> np.array:
        '''
        multiplies the total electric field to the length of the of antenna in each direction
        :return: The voltage at the antenna
        '''
        total_voltage=0
        l = 3.8     #antenna length
        antenna_length = [l,l,0]
        w_ = self.antennna_wave_received()

        w = self.vector_superposition(w_)

        total_voltage = list(map(lambda x : x.real*antenna_length,w))
        # total_voltage = total_voltage.applymap(lambda x: np.sqrt(np.dot(x,x)))

        print('\x1b[1;34mVoltages at antenna:\n\x1b[0m', total_voltage)
        return total_voltage

