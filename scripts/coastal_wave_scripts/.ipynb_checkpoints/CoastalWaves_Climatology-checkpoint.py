print("Starting Script")

import os
os.chdir('/vortexfs1/home/anthony.meza/Atmospheric Rivers and Waves')
print("Changed Directory")

from help_funcs import * 
# from eofs.xarray import Eof
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import importlib
import xarray as xr
# importlib.reload(xr)
import seaborn as sns
import pandas as pd
import cmocean.cm as cm
import netCDF4 as nc
from pathlib import Path
from natsort import natsorted
import matplotlib.pyplot as plt
import gsw

import multiprocessing
from multiprocessing import Pool
from os.path import exists
from pathlib import Path
from natsort import natsorted

print("Loaded Packages")

GLORYS_dir = r"/vortexfs1/home/anthony.meza/GLORYS_data" 
results = [str(result) for result in list(Path(GLORYS_dir).rglob("*.[nN][cC]"))] #get all files
results = natsorted(results) #sort all files 

print("Got Files")

def read_ds_ocn(fname):
    with xr.open_dataset(fname) as ds:
        ds_new = ds[["thetao", "so"]].sel(longitude = slice(-127, -105), 
                                          latitude = slice(20, 50),
                                          depth = slice(0, 250))
        return ds_new
    
print("create first climatology")

ds_climatology = read_ds_ocn(results[0]).isel(time = 0).drop("time")
coords_name = "dayofyear"; coords_size = 365; coords = np.arange(1, coords_size+1)
ds_climatology = ds_climatology.expand_dims(dim = {coords_name: coords_size}).assign_coords({coords_name: coords})
ds_climatology = 0.0 * ds_climatology

ndays = np.zeros(366)

def sum_climatology(fname):
    with read_ds_ocn(fname) as ds:
        dyidx = ds.time.dt.dayofyear.values[0] - 1
        ndays[dyidx]+=1
        ds_climatology.thetao[dyidx, :, :, :] += ds.thetao[0, :, :, :]
        ds_climatology.so[dyidx, :, :, :] += ds.so[0, :, :, :]
j = 0        
print("Starting Read")
nf = len(results)

for fname in results[0:700]:
    if j % 100 == 0:
        print(results[j])
    sum_climatology(fname)
    j+=1
def div_climatology():
    for i in range(0, 366):
        ds_climatology.thetao[i, :, :, :] = ds_climatology.thetao[i, :, :, :] / ndays[i]
        ds_climatology.so[i, :, :, :] = ds_climatology.so[i, :, :, :] / ndays[i]
        
if 0 in ndays:
    print("There was a zero in ndays (something went wrong), need to manually count the days.")    
    ds_climatology.to_netcdf("GLORYS_Daily_Climatology_notnormalized.nc", format="NETCDF4")    
else:
    div_climatology()
    ds_climatology.to_netcdf("GLORYS_Daily_Climatology.nc", format="NETCDF4")
