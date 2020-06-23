import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class visualization(object):
    '''

    '''

    def __init__(self, a: np.array, s: np.array):
        self.a = a
        self.s = s

    def source_antenna_location(self):
        '''
        This function only plot the location of the antenna and sources and the stright pathe from the souce and antenna
        :param a: Array, the location of antenna
        :param s: Array, The location of source/s
        :return: None
        '''

        x_antenna = []
        x_source = []
        for i in range(len(self.a)): x_antenna.append(self.a[i, 0])
        for i in range(len(self.s)): x_source.append(self.s[i, 0])
        # x = x_antenna+x_source

        y_antenna = []
        y_source = []
        for i in range(len(self.a)): y_antenna.append(self.a[i, 1])
        for i in range(len(self.s)): y_source.append(self.s[i, 1])
        # y = y_antenna+y_source

        z_antenna = []
        z_source = []
        for i in range(len(self.a)): z_antenna.append(self.a[i, 2])
        for i in range(len(self.s)): z_source.append(self.s[i, 2])
        # z = z_antenna+z_source

        color_scheme = ['g'] * len(self.a)
        color_scheme = color_scheme + ['r'] * len(self.s)

        # Plotting in 3 dimensions
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle('Please close the picture, when you finished', fontsize=20, color='brown')
        ax = plt.axes(projection='3d')

        plt.title('The overview of the antenna and source coordinates\n\n', fontsize=14)
        ax.scatter3D(x_antenna + x_source,
                     y_antenna + y_source,
                     z_antenna + z_source,
                     color=color_scheme
                     )

        for i in range(len(self.a)):
            for j in range(len(self.s)):
                ax.plot(
                    [x_antenna[i], x_source[j]],
                    [y_antenna[i], y_source[j]],
                    [z_antenna[i], z_source[j]],
                    color='gold')

                ax.text(x_source[j] - 500, y_source[j], z_source[j] + 1000,
                        'Source {}\n'.format(j + 1), size=10, zorder=1,
                        color='r')

        ax.set_xlabel('X ')
        ax.set_ylabel('Y ')
        ax.set_zlabel('Z ')
        plt.show()  # to keep the picture until you close it
        plt.close(fig)  # Free the memory

#
# vec1 = np.array([[ 50. , 50. ,  0.],
#         [ 50. ,100. ,  0.],
#         [109. , 50. ,  0.]])
#
# vec2= np.array([ [  5000.,  10000., 200000.],
#         [-15000.,  10000. ,200000.]])
#
# source_antenna_location(vec1,vec2)