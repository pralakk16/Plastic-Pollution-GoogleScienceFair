
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

locations = pd.read_csv('PlasticMarinePollutionGlobalDatasetCOPY2.csv')
lat = locations['1'].values
lon = locations['2'].values

m = Basemap(projection='moll',lat_0=0, lon_0=0)
x,y = m(lon, lat)
m.drawmapboundary(fill_color='#99ffff')
m.fillcontinents(color='#cc9966',lake_color='#3989cc')
m.scatter(x, y,3,marker='o',color='k')


lat2 = pd.read_csv('unscaled_latitudes.csv')
lon2 = pd.read_csv('unscaled_longitudes.csv')

lat2 = lat2.values
lon2 = lon2.values

m = Basemap(projection='moll',lat_0=0, lon_0=0)
x,y = m(lon2, lat2)
m.drawmapboundary(fill_color='#99ffff')
m.fillcontinents(color='#cc9966',lake_color='#3989cc')
m.scatter(x, y,3,marker='o',color='#8b0000')





