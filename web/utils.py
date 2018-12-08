
from sqlalchemy.orm import sessionmaker
from models import connect_db, PointsOfInterest, ArchitecturalStyles, Architects,POICategories
import pandas as pd
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values
import geopy.distance
import geocoder
import re
from config import BaseConfig


def get_lats_longs(address):
    '''This function uses GeoCoder to retrieve the latitude and longitude for neighbourhoods.
    address to look up
    returns lat and long
    '''
    KEY = BaseConfig.GOOGLE_KEY
    # # get geocode for postal codes
    # initialize your variable to None
    lat_lng_coords = None
    lat=None
    long=None
    max_loops = 10
    curr_loop =0
    # loop until you get the coordinates (sometime dont' get a response)
    lat_lng_coords = None
    while(lat_lng_coords is None and curr_loop < max_loops):
        g = geocoder.google('{}'.format(address), key=KEY)

        lat_lng_coords = g.latlng

        if lat_lng_coords == None:
            time.sleep(1)
        curr_loop += 1
    if lat_lng_coords is not None:
        lat=lat_lng_coords[0]
        long=lat_lng_coords[1]

    return lat,long

def find_dist(df, lat, long):
    avail_points = []

    for ix, row in df.iterrows():
        curr_pt = geopy.distance.geodesic((row['latitude'], row['longitude']), (lat, long)).km

        avail_points.append(curr_pt)
    df['dist_start'] = avail_points
    return df

def find_points_outside_TO(df, fix_address=False, dist=50, starting_lat=43.656287,starting_long= -79.380898):
    ''' Find points more than 50KM from downtown Toronto (Yonge Dundas Square is default) and try to update'''
    df=find_dist(df, starting_lat, starting_long)
    for ix, row in df[df['dist_start']>dist].iterrows():
        print(f"{row['poi_id'], row['name']} is outside of Toronto")
        update_coords(row['poi_id'], fix_address)

def cleanup_address(name, address, logging):
    '''
    Inputs: Stop name and address
    Returns: updated address if appropriate
    '''

    valid_cities = ['Toronto','Scarborough', 'Etobicoke','York']
    valid_provinces = ['ON', 'Ontario']
    new_address = address

    if pd.isna(address):
        return ""
    if address[0] == '0':
        # look for "0 <street" format -- usually first digit
        new_address = row.name + " " +  address[1:]  + ", Toronto ON"
        logging.debug(f"Missing Street address for {address}.  updating address to {new_address}")
        #print(f'try searching on: {new_address}')
    elif len(re.findall('|'.join(valid_cities).lower(), address.lower())) <1:
         # check if missing Toronto and try adding it
        #print(f'{address}: address is missing toronto')
        logging.debug(f"Address {address} is missing a local city -- try appending Toronto ON")
        new_address = address +  ", Toronto ON"
    elif len(re.findall('|'.join(valid_provinces), address)) <1:
        logging.debug(f"Address {address} is missing province.  Try appending ON")
        #print(f"Address {address} is missing province.  Try appending ON")
        new_address = address +  ", ON"
    return new_address
