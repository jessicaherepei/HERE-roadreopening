import pandas as pd
import movingpandas as mpd
import geopandas as gpd
import matplotlib.pyplot as plt 
from shapely.geometry import Point



df2 = pd.read_csv('C:\Developer\HERE-roadreopening\data\construction_1_after.csv')
clean_data1 = df2.drop(['source','country_code','file_name','system_date','tile_id','vehicle_type','sample_date','file_name','vehicle_type','Unnamed: 0'], axis=1) 

print(clean_data1)
# maybe we don't need to drop date?
# clean_data["source"].unique()
