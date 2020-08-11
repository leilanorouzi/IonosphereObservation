import matplotlib.pyplot as plt
import numpy as np
from geometry import  axis_angels

# Read Magnetic field values
import magnetic_field


def b_k_angle(k:list, fnm:str)->list:


  #reading magnetic values
  df = magnetic_field.reading_igrf(fnm)
  b0 = df.loc[0,['xcomponent','ycomponent','zcomponent']]

  # find the angles of the magnetic field respect to the axises.
  # getting the angles between B0 vector and the axis
  theta_b = np.array(axis_angels(b0))

  # getting the angle between wave number vector, k, and the axis
  theta_k = np.array(axis_angels(k))

  # calculating the angle differences
  theta = theta_b-theta_k

  return theta

# wave number vector
k = [2,3,4]

filename = 'Data/Input/igrfwmmData.json'
print(b_k_angle(k,filename))










