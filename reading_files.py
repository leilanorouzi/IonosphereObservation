import pandas as pd
# import numpy as np


def convert_to_int(lists,n):
    # converts a list of string to numbers
    #el: the element in the list
    res = []    # The result to be returned
    for el in lists:
        #To check the element is a number or alist of numbers
        if len(res)<n:
            if (el[0]!='['):
                res.append( float(el))
            else:
                s_el = el[1:-1].split(',')
                res.append([float(s) for s in s_el])

    return res

def read_line(s:str,column_names:list):
    # split every line and space
    l = s.rstrip("\n").split(" ")
    # get numbers
    l = [x for x in l if (x and x != '#')]
    l = convert_to_int(l,len(column_names))

    # convert the new line to a data farme
    l = pd.Series(l, index=column_names)
    return l

def read_antenna(filename):
    # the name of the columns
    column_names = ['x','y','fix','weight','npoint','type','z','ntrench','path','trenchcableadj']

    # To open the file
    read = open(filename, 'r')
    i = 0   # The number of the row of the file

    # An empty data farme to be filled with the antenna values regarding to the column names
    antenna_df = pd.DataFrame(columns = column_names)

    for line in read:
        i+=1
        if 30< i <34 :  # The lines related to the antenna location
            #extracting values from the line
            newline = read_line(line,column_names)

            #To fill the data frame
            antenna_df = antenna_df.append(newline,ignore_index=True)


    read.close()
    print('\x1b[1;33mAntenna variables values:\x1b[0m \n', antenna_df)

    return antenna_df

def read_source(filename):
    # the name of the columns
    column_names = ['x','y','z','A_s','theta_s','f']

    # To open the file
    read = open(filename, 'r')
    i = 0  # The number of the row of the file

    # An empty data farme to be filled with the source values regarding to the column names
    source_df = pd.DataFrame(columns=column_names)

    for line in read:

        newline = line.rstrip("\n")
        if (newline!='' and newline[0] not in ['#','%']):
            newline = read_line(line,column_names)
            source_df = source_df.append(newline,ignore_index=True)
    read.close()
    print('\x1b[1;33mSource variables values:\x1b[0m \n', source_df)
    return source_df

def read_ionosphere(filename):
    # the name of the columns
    column_names = ['Height','N_e','Ne_to_NmF2','hmF2','hmF1','hmE','hmD','foF2','foF1','foE','foD']

    # To open the file
    read = open(filename, 'r')
    i = 0   # The number of the row of the file

    # An empty data farme to be filled with the antenna values regarding to the column names
    iono_df = pd.DataFrame(columns = column_names)

    for line in read:
        i+=1
        if 43< i  :  # The lines related to the antenna location
            #extracting values from the line
            newline = read_line(line,column_names)

            #To fill the data frame
            iono_df = iono_df.append(newline,ignore_index=True)


    read.close()
    print('\x1b[1;33mIonosphere variables values:\x1b[0m \n', iono_df)

    return iono_df


# fn = '../IonosphereObservation/Data/Input/03-tri50_2.txt'
# at = read_antenna(fn)
# print('\x1b[1;33mAntenna variables values:\x1b[0m \n',at)
# at.to_csv('../IonosphereObservation/Data/Input/03-tri50.csv', encoding='utf-8',index= False )

# fn = 'OriginalDocuments/source2.txt'
# sc = read_source(fn)
# print('\x1b[1;36mSource variables values:\x1b[0m \n',sc)
# sc.to_csv('../IonosphereObservation/Data/Input/source2.csv', encoding='utf-8',index= False )




