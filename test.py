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
