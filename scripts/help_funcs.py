# import cdsapi
import numpy as np
import matplotlib.pyplot as plt
import netCDF4
import xarray as xr
from scipy import linalg
from scipy.signal import butter,filtfilt
from math import radians, cos, sin, asin, sqrt

# def getVarUrl(var, time_idx):
#     #maximum time is index 328

#     urlOG = "https://ameza:Ameza1998@my.cmems-du.eu/thredds/dodsC/cmems_mod_glo_phy_my_0.083_P1M-m?longitude[540:1:780],latitude[1320:1:1560],depth[0:1:27],"
#     space_sel = "[0:1:27][1320:1:1560][540:1:780]"
#     time_sel = "time" + time_idx
#     var_sel1 = "thetao" + time_idx + space_sel
#     var_sel2 = "so" + time_idx + space_sel

#     url = urlOG + time_sel +"," + var_sel1 + "," + var_sel2
#     return url 
# c = cdsapi.Client(quiet = True)
def specific_humidity_from_Td(surface_pressure, Td):
    #vapor pressure 
    e = 6.112 * np.exp((17.67 * Td) / (Td + 243.5))
    #specific humididity
    q = (0.622 * e) / (surface_pressure - (0.378 * e))
    return q 

def accessERA5(c, year, path = 'download.grib'): 
    #accessing from https://doi.org/10.24381/cds.f17050d7
    c.retrieve(
        'reanalysis-era5-single-levels-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': [
                'total_precipitation',
            ],
            'year': str(year),
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'time': '00:00',
            'area': [
                50, -135, 30,
                -115,
            ],
            'format': 'grib',
        },
        path)
def normalize(var):
    return (var - np.mean(var))/np.std(var)
def haversine_np(lon1, lat1, lons, lats):
    lon1, lat1, lons, lats = map(np.radians, [lon1, lat1, lons, lats])

    dlon = lons - lon1
    dlat = lats - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lats) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def where_haversine_min(lon1, lat1, lons, lats):
    distances = haversine_np(lon1, lat1, lons, lats)
    return np.unravel_index(np.argmin(distances, axis=None), distances.shape)


def do_EOF_on_xarray(ds):
    ds_mean =  ds.mean(axis= 0) #remvove time average 
    ds_nomean = ds - ds_mean
    
    Y = np.array([a.flatten() for a in ds_nomean])
    Y = Y.T #reshape to space X time 

    not_wet_all = np.isnan(Y)
    not_wet = np.isnan(Y[:, 1]) #land mask 
    Y[not_wet_all] = 0.0 #remove nans 

    u, s, vh  = linalg.svd(Y, full_matrices= False) #do svd 
    
    return Y, (u, s, vh), ds_mean, not_wet
    
def get_latlon_idx(ds, xs, ys):
    #sample along the line 
    LONS, LATS = np.meshgrid(ds.longitude, ds.latitude)
    prev_ind = (np.nan, np.nan)
    lons_list = []
    lats_list = []
    i_ind = []
    j_ind = []
    tot_dist = []
    for (x, y) in zip(xs, ys): 
        new_ind = where_haversine_min(x, y, LONS, LATS)
        is_unique = new_ind !=prev_ind
        if is_unique:
            i_ind.append(new_ind[0])
            j_ind.append(new_ind[1])
            lons_list.append(LONS[new_ind[0], new_ind[1]])
            lats_list.append(LATS[new_ind[0], new_ind[1]])
            tot_dist.append(haversine_np(lons_list[-1], lats_list[-1], xs[-1], ys[-1]))

        prev_ind = new_ind
    return i_ind, j_ind
    #now we have a set of sample pts along a line 
    

def add_doy(ds):
    t = ds["time"]
    doy_original = t.dt.dayofyear
    march_or_later = t.dt.month >= 3
    not_leap_year = ~t.dt.is_leap_year
    doy = doy_original + (not_leap_year & march_or_later)
    ds.coords["doy"] = doy
    return ds
def daily_climatology_leapyears(ds):
    # add day of year as coordinate
    ds = add_doy(ds)
    # daily climatology
    clim_doy = ds.groupby("doy").mean()
    return clim_doy

from scipy.signal import sosfiltfilt, butter
def butter_lowpass_filter(data, cutoff, fs, order):
    # Get the filter coefficients 
    sos = butter(order, cutoff, fs = fs, btype='low', output='sos')
    nt = len(data)
    #pad the data to get a clean transition from Jan to Dec
    extend_data = np.concatenate([data[-60:], data, data[:60]])
    y = sosfiltfilt(sos, extend_data, padtype = None, padlen = 0) #apply filter to obtain a smooth daily climatology
    return y[60:-60]
butter_lowpass = lambda x: butter_lowpass_filter(x,  1/20, 1, 5)

def smooth_daily_climatology(daily_climatology, axis = "dayofyear"):
    smooth_climatology = 0.0 * daily_climatology.compute()
    for key in daily_climatology.keys():
        print(key)
        data = daily_climatology[key].values
        wherenan = np.isnan(data)
        data[wherenan] = 0.0
        axis_num = daily_climatology[key].get_axis_num(axis)
        
        smooth_climatology[key].values = np.apply_along_axis(func1d=butter_lowpass, 
                                                              axis=axis_num, 
                                                              arr=data)
        smooth_climatology[key].values[wherenan] = np.nan
        
    return smooth_climatology

def smooth_daily_climatology_fast(daily_climatology, axis = "dayofyear"):
    smooth_climatology = 0.0 * daily_climatology.compute()
    sos = butter(4, 1/20, btype='low', output='sos', fs = 1)
    nlon = len(daily_climatology.longitude)

    for key in daily_climatology.keys():
        axis_num = daily_climatology[key].get_axis_num(axis)
        for ilon in np.arange(0, nlon):
            print(ilon)
            data = daily_climatology[key].isel(longitude = ilon).values.T
            wherenan = np.isnan(data)
            data[wherenan] = 0.0

            extend_data = np.concatenate([data[..., -60:], data, data[..., :60]], axis = -1)
            #apply filter to obtain a smooth daily climatology
            ysmooth = sosfiltfilt(sos, extend_data, axis = -1, 
                                  padtype = None, padlen = 0)[..., 60:-60]
            ysmooth[wherenan] = np.nan

            smooth_climatology[key].values[..., ilon] = ysmooth.T

    return smooth_climatology

def zonal_average_coastline(var, weights): 
    var_dict = dict()
    var_dict["GC_E"]  = (weights.GC_E * var).sum(dim = "longitude") / (weights.GC_E).sum(dim = "longitude")
    var_dict["GC_W"] = (weights.GC_W * var).sum(dim = "longitude") / (weights.GC_W).sum(dim = "longitude")
    var_dict["GC_C"] = (weights.GC_C * var).sum(dim = "longitude") / (weights.GC_C).sum(dim = "longitude")
    var_dict["EQ"] = (weights.EQ * var).sum(dim = "latitude") / (weights.EQ).sum(dim = "latitude")
    var_dict["SW"] = (weights.SW * var).sum(dim = "latitude") / (weights.SW).sum(dim = "latitude")
    var_dict["NW"] = (weights.NW * var).sum(dim = "longitude") / (weights.NW).sum(dim = "longitude")
    var_dict["COL"] = (weights.COL * var).sum(dim = "longitude") / (weights.COL).sum(dim = "longitude")
    return var_dict

#only pick dry points when the variables varies with depth
#if we do not use this then the above function will overcount the number of actual point available

def zonal_average_coastline_depth(var, weights, wet_points): 
    var_dict = dict()
    var_dict["GC_E"]  = (weights.GC_E * var).sum(dim = "longitude") / (weights.GC_E *wet_points).sum(dim = "longitude")
    var_dict["GC_W"] = (weights.GC_W * var).sum(dim = "longitude") / (weights.GC_W *wet_points).sum(dim = "longitude")
    var_dict["GC_C"] = (weights.GC_C * var).sum(dim = "longitude") / (weights.GC_C *wet_points).sum(dim = "longitude")
    var_dict["EQ"] = (weights.EQ * var).sum(dim = "latitude") / (weights.EQ *wet_points).sum(dim = "latitude")
    var_dict["SW"] = (weights.SW * var).sum(dim = "latitude") / (weights.SW *wet_points).sum(dim = "latitude")
    var_dict["NW"] = (weights.NW * var).sum(dim = "longitude") / (weights.NW *wet_points).sum(dim = "longitude")
    var_dict["COL"] = (weights.COL * var).sum(dim = "longitude") / (weights.COL *wet_points).sum(dim = "longitude")
    return var_dict

def stitch_zonal_average(d):
    var_list = []
    for key in ["EQ", "COL", "SW", "GC_E", "GC_C", "GC_W", "NW"]:
        d[key] = d[key].compute()
        if key == "EQ":
            var = d[key][~np.isnan(d[key])].compute().values
        else:
            if key == "GC_C":
                print(key)
                var = d[key][~np.isnan(d[key])].compute().values[::-1]
            elif key == "SW":
                print(key)
                var = d[key][~np.isnan(d[key])].compute().values[::-1]
            else:
                var = d[key][~np.isnan(d[key])].compute().values

        var_list = np.concatenate((var_list, var))
    return var_list 
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

def stich_zonal_average_xr(ds_dict, longitudes, latitudes, cum_distance):
    var_list = []
    n_dist1 = 0
    for key in ["EQ", "COL", "SW", "GC_E", "GC_C", "GC_W", "NW"]:
        if key == "EQ":
            var = ds_dict[key].isel(longitude = np.where(~np.isnan(longitudes[key]))[0])
            var = var.rename({'longitude': 'distance'})
        else:
            if key == "GC_C":
                var = ds_dict[key].isel(latitude = np.where(~np.isnan(latitudes[key]))[0]).isel(latitude=slice(None, None, -1))
                var = var.rename({'latitude': 'distance'})
            elif key == "SW":
                var = ds_dict[key].isel(longitude = np.where(~np.isnan(longitudes[key]))[0]).isel(longitude=slice(None, None, -1))
                var = var.rename({'longitude': 'distance'})
            else:
                var = ds_dict[key].isel(latitude = np.where(~np.isnan(latitudes[key]))[0])
                var = var.rename({'latitude': 'distance'})

        n_dist2 = len(var.distance)
        var = var.assign_coords({"distance": cum_distance[n_dist1:n_dist1+n_dist2]})
        n_dist1 = n_dist2 + n_dist1
        var_list = var_list + [var]
    return var_list

# window = tukey(len(zos.distance), 0.05)
def filter_negative_wavenumbers(data, dt=1, dx=1):
    
    #legnth of dimensions
    nt, nx = data.shape

    fft_vals = fft2(data)
    
    #wave definition is exp(i(k*x-omega*t)) but FFT definition exp(-ikx)
    #so change sign
    omega = fftfreq(nt, 1)
    k = fftfreq(nx, 100); k =-1*k

    fft_shift = fftshift(fft_vals)
    k_grid, omega_grid = np.meshgrid(fftshift(k), fftshift(omega))

    #filter regularly gridded wavenumber and frequency
    fft_shift[omega_grid < 0.0] = 0*fft_shift[omega_grid < 0.0]
    fft_shift[omega_grid > 0.0] = 2*fft_shift[omega_grid > 0.0]
    
    fft_shift[k_grid < 0.0] = 0*fft_shift[k_grid < 0.0]
    fft_shift[k_grid > 0.0] = 2*fft_shift[k_grid > 0.0]


    return np.real(ifft2(ifftshift(fft_shift))), (omega_grid, k_grid, fft_shift)

def get_dispersion_data(data, dt=1, dx=1):
    
    #legnth of dimensions
    nt, nx = data.shape

    fft_vals = fft2(data)
    
    #wave definition is exp(i(k*x-omega*t)) but FFT definition exp(-ikx)
    #so change sign
    omega = fftfreq(nt, 1)
    k = fftfreq(nx, 100); k =-1*k #in hundereds of kilomoters

    fft_shift = fftshift(fft_vals)
    k_grid, omega_grid = np.meshgrid(fftshift(k), fftshift(omega))

    #filter regularly gridded wavenumber and frequency
    fft_shift[omega_grid < 0.0] = 0*fft_shift[omega_grid < 0.0]
    fft_shift[omega_grid > 0.0] = 2*fft_shift[omega_grid > 0.0]
    return omega_grid, k_grid, fft_shift
