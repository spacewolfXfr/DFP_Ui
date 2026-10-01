#!/usr/bin/env python

import argparse
import datetime
import dateutil.parser

import pandas as pd

from .AirportHelpers import get_airport
from .FlightExtraction import NM_TO_METERS

from pitot.geodesy import destination

from traffic.core import Traffic
from traffic.data import opensky


def datetime_from_epoch_or_iso8601(timestr: str) -> datetime.datetime:
    """Convert a string representing a timestamp in seconds since epoch or an ISO 8601 formatted string to a datetime object."""
    try:
        # Try to interpret the string as an integer (epoch time)
        timestamp = int(timestr)
        return datetime.datetime.fromtimestamp(timestamp)
    except ValueError:
        # If it fails, try to parse it as an ISO 8601 string
        return dateutil.parser.isoparse(timestr)

def main():
    parser = argparse.ArgumentParser(prog="Traffic Fetcher", description="Using the OpenSky API, fetch traffic data for a given airport and save it to a parquet file.")
    parser.add_argument('airport', help="ICAO code of the airport to fetch traffic for")
    parser.add_argument('start', type=str, help="Start timestamp (in seconds since epoch, or using the ISO 8601 format) for the traffic data to fetch.")
    parser.add_argument('--end', type=str, default=None, help="End timestamp (in seconds since epoch, or using the ISO 8601 format) for the traffic data to fetch. If not provided, it is set to 24 hours after the start time.")
    parser.add_argument('--duration', type=float, default=24*60, help="Duration in minutes for which to fetch traffic data. Default is 24 hours. Ignored if --end is provided.")
    parser.add_argument('--output', help="Path to the output parquet file. Default to [ICAO]_[start]_[end].parquet", default=None)
    parser.add_argument('--maxdist', type=float, default=10000.0, help="Maximum distance (in meters) from the airport to consider for fetching traffic data. Default is 10000 meters.")
    parser.add_argument('-NM', '--nautical_miles', action='store_true', help="If set, the --maxdist argument is interpreted in nautical miles instead of meters.")

    args = parser.parse_args()
    
    airport_icao = args.airport
    airport = get_airport(airport_icao)
    print(f"Airport {airport.name} ({airport.icao}) located at lat {airport.latitude}, lon {airport.longitude}")
    # display_airport(airport)
    
    datetime_start = datetime_from_epoch_or_iso8601(args.start)
        
    if args.end is None:
        datetime_end = datetime_start + datetime.timedelta(minutes=args.duration)
    else:
        datetime_end = datetime_from_epoch_or_iso8601(args.end)
        
    if args.nautical_miles:
        args.maxdist *= NM_TO_METERS  # Convert nautical miles to meters
        
    north_bound = destination(airport.latitude, airport.longitude, 0, args.maxdist)[0]
    south_bound = destination(airport.latitude, airport.longitude, 180, args.maxdist)[0]
    east_bound  = destination(airport.latitude, airport.longitude, 90, args.maxdist)[1]
    west_bound  = destination(airport.latitude, airport.longitude, 270, args.maxdist)[1]

    bounds = (west_bound, south_bound, east_bound, north_bound)

    result = opensky.history(datetime_start, datetime_end,
                             bounds=bounds,
                             airport=airport.icao, time_buffer=pd.Timedelta(hours=1))
    if result is None:
        print("No traffic data fetched. Please check the airport ICAO code and the time range.")
        return
    else:
        output = args.output
        if output is None:
            output = f"{airport_icao}_{datetime_start.strftime('%Y%m%dT%H%M%S')}_{datetime_end.strftime('%Y%m%dT%H%M%S')}.parquet"
        result.assign_id().eval().to_parquet(output)
        print(f"Traffic data for airport {airport_icao} saved to {output}")

if __name__ == "__main__":
    main()
