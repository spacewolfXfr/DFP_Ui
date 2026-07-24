import typing
import pathlib

import pandas as pd

import traffic
from traffic.core import Traffic,Flight
from traffic.core.structure import Airport
from traffic.data.basic.airports import Airports

from .FlightExtraction import flight_landing,flight_departing
from DubinsFleetPlanner.thread_estimator import thread_count


def iter_flightdata_by_day(input_file: pathlib.Path, time_column:str='timestamp', id_column:str='flight_id') -> typing.Generator[Traffic, None, None]:
    """Load from the given Parquet file the timestamps and flight ids and group them by day. Produce a generator
    yielding all columns from the Parquet file on a day by day basis (flights may start the day before or end the day after, the
    associated points are provided).
    
    
    A flight belong to a day if its first or last timestamp belong to it, we assume flights span less than 24h.

    Args:
        input_file (pathlib.Path): Path to the Parquet file containing the flight data.
        time_column (str, optional): Column name for the timestamp. Defaults to 'timestamp'.
        id_column (str, optional): Column name for the flight ID. Defaults to 'flight_id'.

    Returns:
        typing.Generator[Traffic, None, None]: Generator for each day's flight data as a Traffic object.

    Yields:
        Iterator[typing.Generator[Traffic, None, None]]: Iterator yielding Traffic objects for each day's flight data.
    """
    try:
        df = pd.read_parquet(input_file,columns=[time_column,id_column])
    except Exception as e:
        print(f"Error reading Parquet file: {e}")
        print("Assuming it was because there is not id... Falling back to icao24")
        id_column = 'icao24'
        df = pd.read_parquet(input_file,columns=[time_column,id_column])

    groups = df.groupby(id_column,sort=False)
    min_vals = groups.min()
    max_vals = groups.max()
    min_vals['flight_id_col'] = min_vals.index
    max_vals['flight_id_col'] = max_vals.index
    
    day_grouper = pd.Grouper(key=time_column, freq='D')
    start_groups = min_vals.groupby(day_grouper,sort=True)
    end_groups = max_vals.groupby(day_grouper,sort=True)
    
    for key in start_groups.groups.keys():
        starting = start_groups.get_group(key)
        ending = end_groups.get_group(key)
        interval = starting.merge(ending, on=id_column, how='outer', suffixes=('_start', '_end')).drop(columns=['flight_id_col_start', 'flight_id_col_end'])
        
        yield Traffic(pd.read_parquet(input_file,filters=[(id_column, 'in', interval.index)]))
    

def filter_traffic(traffic:Traffic,ICAO_set:typing.Iterable[str]|Airports,maxdist:float=10000.) -> typing.Optional[Traffic]:    
    def filter_fun(flight:Flight) -> bool:
        landing = flight_landing(flight,ICAO_set,maxdist)
        departing = flight_departing(flight,ICAO_set,maxdist)
        flight.data['landing_airport'] = landing.icao if landing is not None else None
        flight.data['departing_airport'] = departing.icao if departing is not None else None
        return landing is not None or departing is not None

    good_traffic = traffic.pipe(filter_fun).eval(thread_count())
    if good_traffic is None or len(good_traffic) == 0:
        return None
    
    if isinstance(good_traffic,pd.DataFrame):
        good_traffic = Traffic(good_traffic)
        
    # enhanced_traffic = good_traffic.compute_xy()
    enhanced_traffic:Traffic = good_traffic.cumulative_distance(compute_gs=False,compute_track=True).eval(thread_count()) #type: ignore
    
    return enhanced_traffic


def test():
    from traffic.data.samples import quickstart
    dataset = quickstart
    
    filtered = filter_traffic(dataset,["LFPG","LFPO"])
    filtered.line_geo().show()
