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
