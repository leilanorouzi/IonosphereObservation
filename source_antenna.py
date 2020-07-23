import pandas as pd
from source import Source
from antenna import Antenna
import geometry
from parameters import *


class SourceAntenna(Source, Antenna):
    '''

    :param radar_fn: The path address of the input file of the antenna
    :type radar_fn: str
    :param source_fn: The path address of the input file of the source
    :type source_fn: str
    :param dipole: If the source is a dipole or not. default value is False
    :type dipole: bool
    '''
    def __init__(self,source_fn:str , radar_fn:str , dipole=False):

        # To pass values to parent classes
        Source.__init__(self, source_fn, dipole)
        Antenna.__init__(self,radar_fn)

        # Source to antenna distances
        self.distance, self.path_vec = self.multi_dist()
        print("\x1b[1;31mDistance:\n\x1b[0m", self.distance,
              "\n\x1b[1;31mVector:\n\x1b[0m", self.path_vec )

    def multi_dist(self)->(pd.DataFrame , pd.DataFrame):
        '''
        This function calculates the distance between source/s and antenna/s and also returns a vector of
        source-antenna for every set of source-antenna.

        :returns:
            - dist_arr: the distances from the source to the antenna
            - sa_vec:  the vector from source to the antenna in Cartesian coordinate system (x,y,z).
                        each elements are a list of vector components
        :rtype: pandas.Dataframe
        '''

        dist_arr = pd.DataFrame(np.zeros((self.radar_n, self.source_n)),
                                columns=self.s_columns)
        sa_vec = pd.DataFrame(np.zeros((self.radar_n, self.source_n)),
                              columns=self.s_columns)

        for i in range(self.radar_n):
            for j in range(self.source_n):
                dist_arr.iloc[i, j] = geometry.dist(self.source_location[j, :], self.antenna_location[i, :])
                sa_vec.iloc[[i], j] = pd.Series([-self.source_location[j, :] + self.antenna_location[i, :]], index=[i])
        return dist_arr, sa_vec

    def dipole_transmitter(self):
        '''
        For a far feild a radiation pattern whose electric field of a half-wave dipole antenna  is given by
        https://en.wikipedia.org/wiki/Dipole_antenna#Short_dipole

        we assume that all sources are a dipole which has an angle of theta between
         the direction of dipole and the z direction.
        If you have another assumption you can add it to define theta for each source


        :return:
            - phase: phase part of the generated wave related to the position of the antenna respect to the reference frame
            - e_theta: Generated wave form
        :rtype: Dataframe
        '''

        # the data frame to find the angle of eache dipoles respect to the ray path
        theta = pd.DataFrame(np.zeros([self.radar_n, self.source_n]), columns=self.s_columns)
        for i_s in range(self.source_n):
            for i_a in range(self.radar_n):
                s = self.source_location[i_s]  # The location of sources
                a = self.antenna_location[i_a]  # The location of antennas

                # DE = DC - CE  = DC- AC sin(self.theta_x)
                #               = S_y - S_z * tan(self.theta_z) * sin(self.theta_x)
                dy = s[1] - s[2] * np.tan(self.theta_z) * np.sin(self.theta_x)

                # AK = OD - AE  = OD - AC * cos(self.theta_x)
                #               = S_x - S_z * tan(theta_z) * cos(self.theta_x)
                dx = s[0] - s[2] * np.tan(self.theta_z) * np.cos(self.theta_x)

                p_z0 = np.array([dx, dy, 0])  # The cros section of dipole line and XY plane
                print('XY plane intesection:\n', p_z0)
                dipole_line = p_z0 - s  # Dipole vector

                print('Dipole vector:\n', dipole_line)

                # the angle between the dipole direction and ray path
                # theta = arcsin( a.b / |a| |b|)

                # inner product of dipole vector and the source-antenna vector
                arcsin = np.inner(dipole_line, self.path_vec.iloc[i_a, i_s])
                # |a|= √ (a.a)
                a_len = np.sum(np.square(dipole_line))
                # |b|= √ (b.b)
                b_len = np.sum(np.square(self.path_vec.iloc[i_a, i_s]))
                theta.iloc[i_a, i_s] = np.arcsin(arcsin / np.sqrt(a_len * b_len))

        e_theta = eta * self.I0 * np.cos(0.5 * np.pi * np.cos(theta)) \
                  / (2 * np.pi * np.sin(theta) * self.distance)

        phase = np.cos(0.5 * np.pi * np.cos(theta))/ np.sin(theta)

        print('\x1b[1;31mAangles of dipoles:\n\x1b[0m', np.degrees(theta))

        return e_theta, phase


