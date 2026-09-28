import typing

import numpy as np

import pyproj
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

from traffic.core.structure import Airport
from traffic.data import airports
from traffic.data.basic.airports import Airports

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from cartes.crs import UTM,Lambert93

def get_airport(icao:str) -> Airport:
    return airports[icao]

def get_airports(ICAO_set:typing.Iterable[str]|Airports) -> Airports:
    if isinstance(ICAO_set,Airports):
        return ICAO_set
    data = airports.data
    selected = data[data["icao"].isin(ICAO_set)]
    return Airports(selected)

def get_runway(port:Airport,rw_name:str):
    try:
        return port.runways.data.query(f"name=='{rw_name}'").iloc[0] # type: ignore
    except (ValueError, IndexError):
        print(f"WARNING: Runway {rw_name} not found for airport {port}")
        print(port.runways.data)
        return None

def get_other_runway_name(rw_name:str) -> str:
    """Given a runway name, try to find its name in the other direction.
    
    The rules are as follows:
    - The name in split in two parts: a number (the first two characters) and an optional letter (the third character, if it exists)
    - The number is converted to the other direction by adding 18 if it is less than 18, and substracting 18 otherwise. If the result is 0, it is replaced by 36.
    - The letter, if it exists, is converted as follows: C is unchanged, R becomes L and L becomes R.

    Args:
        rw_name (str): Runway name

    Returns:
        str: Its guessed name in the other direction
    """
    rw_num = int(rw_name[:2])
    rw_numbis = rw_num + (18 if rw_num < 18 else -18)
    if rw_numbis == 0:
        rw_numbis = 36
    if len(rw_name) == 2:
        return f"{rw_numbis:02}"
    else:
        if rw_name[2] == 'C':
            return f"{rw_numbis:02}C"
        elif rw_name[2] == 'R':
            return f"{rw_numbis:02}L"
        else:
            return f"{rw_numbis:02}R"

def get_airport_latlon_transformer(port:Airport) -> pyproj.Transformer:
    lat = port.latitude
    lon = port.longitude
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=lon,
            south_lat_degree=lat,
            east_lon_degree=lon,
            north_lat_degree=lat,
        ),
    )
    utm_crs = pyproj.CRS.from_epsg(utm_crs_list[0].code)
    
    return pyproj.Transformer.from_crs(
        "EPSG:4326",   # WGS84 (lat, lon)
        utm_crs,       # UTM (x, y)
    )
    
def display_airport(airport:Airport) -> None:
    wgs84_to_local = get_airport_latlon_transformer(airport)
    dest_proj = UTM(zone=wgs84_to_local.target_crs.to_dict()['zone'], southern_hemisphere=airport.latitude < 0)
    fig,ax = plt.subplots(figsize=(10, 10),subplot_kw={"projection": Lambert93()})
    ax.set_title(f"Airport {airport.name} ({airport.icao})")
    airport.plot(ax=ax, footprint=True,
                runways=dict(color="#f58518"),  # update default parameters
                labels=dict(fontsize=12))
    ax.spines['geo'].set_visible(False)
        
    plt.show()