if __name__ == "__main__": 
    import os
    os.chdir('/vortexfs1/home/anthony.meza/Atmospheric Rivers and Waves/scripts')
    plotsdir = lambda x="": "/vortexfs1/home/anthony.meza/Atmospheric Rivers and Waves/plots/" + x
    GLORYS_dir = lambda x="": "/vortexfs1/home/anthony.meza/GLORYS_data" + x
    GLORYS_data_dir = lambda x="": "/vortexfs1/home/anthony.meza/Atmospheric Rivers and Waves/GLORYS_processed/" + x

    # from help_funcs import * 
    import xarray as xr
    import pandas as pd
    import netCDF4 as nc
    from pathlib import Path
    from natsort import natsorted
    import matplotlib.pyplot as plt
    import gc
    import os 
    from os.path import exists
    import dask_labextension

    from dask_jobqueue import SLURMCluster  # setup dask cluster 
    cluster = SLURMCluster(
        cores=36,
        processes=1,
        memory='192GB',
        walltime='02:00:00',
        queue='compute',
        interface='ib0')
    print(cluster.job_script())
    cluster.scale(jobs=4)
    from dask.distributed import Client
    client = Client(cluster)
    client

    results = [str(result) for result in list(Path(GLORYS_dir()).rglob("*.[nN][cC]"))] #get all files
    results = natsorted(results) #sort all files 

    years = natsorted(list(set([result[41:45] for result in results])))
    months = natsorted(['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'])

    def _preprocess(ds):
        return ds[["thetao", "zos"]].sel(latitude = slice(-2, 60), 
                                         longitude = slice(-150, -75)).sel(depth = [0, 5, 20, 50], method = "nearest")

    ds = xr.open_mfdataset(
            results,
            data_vars="minimal",
            coords="minimal",
            compat="override",
            preprocess=_preprocess,
            parallel=True,
            chunks={"latitude":-1, "longitude":-1, "time":-1},
            engine="netcdf4")
    #remove leap years to get a smooth climatology
    ds = ds.convert_calendar('noleap')
    ds["time"] = ds.indexes['time'].to_datetimeindex()

    #save the dataset before processing 
    ds.to_netcdf(GLORYS_data_dir("GLORYS_NE_PAC.nc"),
                 mode = "w", format = "NETCDF4", 
                 engine = "netcdf4", compute = True)
    print("Saved the full field")
    ds_climatology = ds.groupby("time.dayofyear").mean("time")
    ds_climatology.to_netcdf(GLORYS_data_dir("GLORYS_SFC_Climatology.nc"),
                 mode = "w", format = "NETCDF4", 
                 engine = "netcdf4", compute = True)
    print("Saved the climatology field")
    #reload the optimized GLORYS files
    ds = xr.open_mfdataset(GLORYS_data_dir("GLORYS_NE_PAC.nc"), 
                                data_vars="minimal",
                                coords="minimal",
                                compat="override",
                                parallel=True,
                                chunks={"longitude": 100, "latitude":100, "time":-1},
                                engine="netcdf4")
    ds_climatology = xr.open_mfdataset(GLORYS_data_dir("GLORYS_SFC_Climatology.nc"), 
                                parallel=True,
                                chunks={"longitude": 100, "latitude":100},
                                engine="netcdf4")
    print("Reloading full field and climatology")
    gc.collect()
    ds_anomalies = ds.groupby("time.dayofyear") - ds_climatology
    ds_anomalies.to_netcdf(GLORYS_data_dir("GLORYS_NE_PAC_Anomalies.nc"),
                 mode = "w", format = "NETCDF4", 
                 engine = "netcdf4", compute = True)
    print("Saved Anomalies")
