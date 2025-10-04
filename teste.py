import numpy as np
import xarray as xr

swot_path = "SWOT_L2_LR_SSH_Expert_008_497_20240101T000705_20240101T005833_PIC0_01.nc"
# modis_path = "data/AQUA_MODIS.20240101.L3b.DAY.AT202.nc"  # Comentado pois não temos este arquivo

ds_swot = xr.open_dataset(swot_path)
# ds_modis = xr.open_dataset(modis_path)  # Comentado pois não temos este arquivo

print("---- SWOT ----")
print(ds_swot)
# print("\n---- MODIS ----")
# print(ds_modis)

# Get coordinate names (lat/lon may be different)
print("SWOT coords:", list(ds_swot.coords))
# print("MODIS coords:", list(ds_modis.coords))


# Quick bounding box function
def bbox(ds, lat_name=None, lon_name=None):
    if lat_name is None:
        lat_name = [c for c in ds.coords if "lat" in c.lower()][0]
    if lon_name is None:
        lon_name = [c for c in ds.coords if "lon" in c.lower() or "lon" in c][0]
    lats = ds[lat_name].values
    lons = ds[lon_name].values
    return np.nanmin(lats), np.nanmax(lats), np.nanmin(lons), np.nanmax(lons)


swot_bbox = bbox(ds_swot)
# modis_bbox = bbox(ds_modis)  # Comentado pois não temos este arquivo
print("SWOT bbox (minlat,maxlat,minlon,maxlon):", swot_bbox)
# print("MODIS bbox:", modis_bbox)


# Time ranges
def time_range(ds):
    # Procura primeiro nas coordenadas, depois nas variáveis de dados
    tname = None
    for c in ds.coords:
        if "time" in c.lower():
            tname = c
            break

    if tname is None:
        for v in ds.data_vars:
            if "time" in v.lower():
                tname = v
                break

    if tname is None:
        return "Time variable not found"

    t = ds[tname].values
    return np.min(t), np.max(t)


print("SWOT time range:", time_range(ds_swot))
# print("MODIS time range:", time_range(ds_modis))  # Comentado pois não temos este arquivo
