import pandas as pd

def convert_to_int(lists,n):
    '''
    converts a list of string to numbers

    :param lists: the element in the list
    :param n:
    :return:
    '''

    # The result to be returned
    res = []

    for el in lists:
        #To check the element is a number or a list of numbers
        if len(res)<n:
            if (el[0]!='['):
                res.append( float(el))
            else:
                s_el = el[1:-1].split(',')
                res.append([float(s) for s in s_el])

    return res

def read_line(s:str,column_names:list)->pd.Series:
    '''

    :param s: the line
    :param column_names: a list of column's name
    :return: a series of values
    '''
    # split every line and space
    l = s.rstrip("\n").split(" ")
    # get numbers
    l = [x for x in l if (x and x != '#')]
    l = convert_to_int(l,len(column_names))

    # convert the new line to a data farme
    l = pd.Series(l, index=column_names)
    return l

# def read_ionosphere(filename):
#     # the name of the columns
#     column_names = ['Height','N_e','Ne_to_NmF2','hmF2','hmF1','hmE','hmD','foF2','foF1','foE','foD']
#
#     # To open the file
#     read = open(filename, 'r')
#     i = 0   # The number of the row of the file
#
#     # An empty data farme to be filled with the antenna values regarding to the column names
#     iono_df = pd.DataFrame(columns = column_names)
#
#     for line in read:
#         i+=1
#         if 43< i  :  # The lines related to the antenna location
#             #extracting values from the line
#             newline = read_line(line,column_names)
#
#             #To fill the data frame
#             iono_df = iono_df.append(newline,ignore_index=True)
#
#
#     read.close()
#     print('\x1b[1;33mIonosphere variables values:\x1b[0m \n', iono_df)
#
#     return iono_df