# import numpy as np
import pandas as pd
import reading_files


class Antenna(object):
    '''
    :param radar_fn: Antenna input file path
    :type radar_fn: str
    '''
    def __init__(self,radar_fn):

        self.filename = radar_fn

        self.antenna_charc = self.read_antenna()
        self.antenna_location = self.antenna_charc.loc[:, ['x', 'y', 'z']].values  # Taking coordination
        self.radar_n = len(self.antenna_charc)

    def read_antenna(self):
        '''
        :param filename: The file address name of the antenna properties
        :return: A data frame of the antenna parameters
        '''
        # the name of the columns
        column_names = ['x', 'y', 'fix', 'weight', 'npoint', 'type', 'z', 'ntrench', 'path', 'trenchcableadj']

        # To open the file
        read = open(self.filename, 'r')
        i = 0  # The number of the row of the file

        # An empty dataframe to be filled with the antenna values regarding to the column names
        antenna_df = pd.DataFrame(columns=column_names)

        for line in read:
            i += 1
            if 30 < i < 34:  # The lines related to the antenna location
                # extracting values from the line
                newline = reading_files.read_line(line, column_names)

                # To fill the data frame
                antenna_df = antenna_df.append(newline, ignore_index=True)

        read.close()
        print('\x1b[1;33mAntenna variables values:\x1b[0m \n', antenna_df)

        return antenna_df
