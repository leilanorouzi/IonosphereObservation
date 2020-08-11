# https://earth-planets-space.springeropen.com/articles/10.1186/s40623-015-0228-9
# https://www.ngdc.noaa.gov/IAGA/vmod/igrf.html
# https://www.ngdc.noaa.gov/geomag/models.shtml
# https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml#igrfgrid
# F - Total Intensity of the geomagnetic field
# H - Horizontal Intensity of the geomagnetic field
# X - North Component of the geomagnetic field
# Y - East Component of the geomagnetic field
# Z - Vertical Component of the geomagnetic field
# I (DIP) - Geomagnetic Inclination
# D (DEC) - Geomagnetic Declination (Magnetic Variation)
# MF = Main Field
# SV = Secular Variation

import json
import pandas as pd

def reading_igrf(filename):

  with open(filename) as f:
    data = json.load(f)

  df = pd.DataFrame.from_dict(data['result'])
  # units = pd.DataFrame.from_dict(data['units'])
  for i in range(len(df.keys())):
    print(df.keys()[i], df.iloc[0,i])

  return df