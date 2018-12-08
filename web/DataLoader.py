#!/usr/bin/env python
# coding: utf-8

# # Data Loader
# * loads csv file, expected to have set structure of columns even if not all used
# * cleans data:
# *   1) find and remove duplicates
# *   2) clean up addresses
# *   3) get lats/longs
# *   4) calculate additional columns
# * uploads to db


from sqlalchemy.orm import sessionmaker
from models import connect_db, PointsOfInterest, ArchitecturalStyles, Architects,POICategories
import pandas as pd
import re

from config import BaseConfig
from utils import *


import logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename='../logs/initdataload.log',level=logging.INFO)


df_to_db_map={
    'name':'name',
    'build_year':'build_year'   ,
    'demolished_year' :'demolished_year',
    'address' :'address' ,
    'external_url':'external_url',
    'details': 'details',
    'image_url':'image_url',
    'heritage_status':'heritage_status',
    'current_use use':'current_use',
    'poi_type':'poi_type',
    'source': 'source',
    'cleaned_year': 'build_year_clean',
    'build_decade': 'build_decade',
    'poi_type_simple': 'poi_type_simple',
    'latitude': 'latitude',
    'longitude': 'longitude'
}


def make_simple_poi(org_type):
    if org_type == 'Plaque':
        return org_type
    elif org_type == 'Monument':
        return 'Art'
    else:
        return 'Building'


def clean_build_year(year):
    if pd.isna(year) or year == None or len(year) < 4:
        return ''
    strip_words = ['unknown', 'circa ', 'abt ', 'about']
    for word in strip_words:
        year=year.replace(word, '')
    return year[0:4]

# try to find points outside of Toronto
def find_points_outside_TO(df, fix_address=False, dist=50, starting_lat=43.656287,starting_long= -79.380898):
    ''' Find points more than 50KM from downtown Toronto (Yonge Dundas Square is default) and try to update'''
    df=find_dist(df, starting_lat, starting_long)
    for ix, row in df[df['dist_start']>dist].iterrows():
        #print(f"{row['poi_id'], row['name']} is outside of Toronto")
        logging.debug(f"{row['poi_id'], row['name']} is outside of Toronto")
        #update_coords(row['poi_id'], fix_address)
        lat, long = get_lats_longs(row)
        df_poi.loc[index, 'latitude']= lat
        df_poi.loc[index, 'longitude']= long
    return df


def add_features(df):
    df['cleaned_year']=df['build_year'].apply(lambda x: clean_build_year(x))
    df['cleaned_year']=pd.to_numeric(df['cleaned_year'],errors='coerce',downcast='integer')
    df['build_decade']= df['cleaned_year'].apply(lambda x: x//10*10 )
    df['poi_type_simple'] = df['poi_type'].apply(lambda x: make_simple_poi(x))
    return df


def load_init_data():
    df_poi = pd.read_csv('init_data/pois.csv' )
    df_architects = pd.read_csv('init_data/architects.csv' )
    df_cats = pd.read_csv('init_data/poi_cats.csv' )
    df_styles= pd.read_csv('init_data/architectural_styles.csv' )
    return df_poi, df_architects, df_cats, df_styles


def data_clean_up(df_poi, cols_check_dups = ['name', 'address','source','external_url']):
    '''
    Load from CSV
    Drop duplicates
    Clean up address
    Find missing lat/long coords
    Find spots outside Toronto and
    add new features
    '''
    #df_poi=load_init_data()
    df_poi=df_poi.drop_duplicates(subset=cols_check_dups, keep='last')
    df_poi['address']=df_poi.apply(lambda row: cleanup_address(row['name'], row['address'], logging), axis=1)
    for index, row in df_poi[pd.isna(df_poi['latitude']) | pd.isna(df_poi['longitude'])].iterrows():
        lat, long = get_lats_longs(row)
        df_poi.loc[index, 'latitude']= lat
        df_poi.loc[index, 'longitude']= long
    df_poi=find_points_outside_TO(df_poi)
    df_poi = add_features(df_poi)
    return df_poi


def save_to_database_ORM(session, df):
    '''
    Saves scraped data to database using SqlAlchemy ORM
    Updates three tables: points_of_interest, archtectural_styles, architects
    The relationship between these tables is defined in models.py, so it automatically populates the poi_id column
    in the child tables with the poi_id of the main entry
    '''

    for index, row in df.iterrows():

        poi_dict ={df_to_db_map[k]:v for k, v in row.items() if k in df_to_db_map.keys() and not pd.isnull(v)}
        #poi_dict['source']= site_root
        poi = PointsOfInterest(**poi_dict )
        old_poi_id =row['poi_id']

        # define style
        for ix2, astyle in df_styles[df_styles['poi_id']==old_poi_id].iterrows():
            #tyle=ArchitecturalStyles(style=row['Style'])
            style=ArchitecturalStyles(style=astyle['style'])
            poi.styles.append(style)

        for ix2, acat in df_cats[df_cats['poi_id']==old_poi_id].iterrows():
            cat = POICategories(category =acat['category'])
            poi.categories.append(cat)

        # architects (can be multiple)
        prev_company=""
        for ix2, anarct in df_architects[df_architects['poi_id']==old_poi_id].iterrows():
            if anarct['architect_name'] != prev_company and not 'Also see' in anarct['architect_name']:
                architect = Architects(architect_name= anarct['architect_name'].replace("'","''"))
                poi.architects.append(architect)
                prev_company=anarct['architect_name']
                #print (anarct['architect_name'])

        session.add(poi)
        session.commit()

# if BaseConfig.POPULATE_DB:
#     logging.debug("Starting population of database")
#     df_poi, df_architects, df_cats, df_styles=load_init_data()
#     df_poi=data_clean_up(df_poi)
#     df_poi.head()
#     db=connect_db() #establish connection
#     Session = sessionmaker(bind=db)
#     session = Session()
#     save_to_database_ORM(session, df_poi[df_poi['poi_id']!=8])
#     print(session.query(PointsOfInterest).count())
#
# else:
#     print("The POPULATE_DB flag is false... exiting without action")
