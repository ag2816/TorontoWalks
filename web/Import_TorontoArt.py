from sqlalchemy.orm import sessionmaker
from models import connect_db, PointsOfInterest, ArchitecturalStyles, Architects,POICategories
import pandas as pd
import re

from utils import *
from DataLoader import *


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


        # # define style
        # for ix2, astyle in df_styles[df_styles['poi_id']==old_poi_id].iterrows():
        #     #tyle=ArchitecturalStyles(style=row['Style'])
        #     style=ArchitecturalStyles(style=astyle['style'])
        #     poi.styles.append(style)
        #
        # for ix2, acat in df_cats[df_cats['poi_id']==old_poi_id].iterrows():
        #     cat = POICategories(category =acat['category'])
        #     poi.categories.append(cat)

        # architects (can be multiple)
    #    for ix2, anarct in df_architects[df_architects['poi_id']==old_poi_id].iterrows():
        architect = Architects(architect_name= row['artist'])
        poi.architects.append(architect)
        # print(row['artist'])
        # print(poi.architects)
        session.add(poi)
        session.commit()

if __name__ == "__main__":
    df = pd.read_csv('data/public_art.csv')
    #df.head()
    df=data_clean_up(df,cols_check_dups = ['name', 'address','source'])
    #df.head()
    db=connect_db() #establish connection
    Session = sessionmaker(bind=db)
    session = Session()
    print(session.query(PointsOfInterest).count())
    save_to_database_ORM(session, df)
    print(session.query(PointsOfInterest).count())
