from sqlalchemy.orm import sessionmaker
from models import connect_db, PointsOfInterest, ArchitecturalStyles, Architects,POICategories
import pandas as pd
import re

from config import BaseConfig
from utils import *
from DataLoader import *

import logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename='../logs/initdataload.log',level=logging.INFO)

if __name__ == "__main__":
    if BaseConfig.POPULATE_DB:
        logging.debug("Starting population of database")
        df_poi, df_architects, df_cats, df_styles=load_init_data()
        df_poi=data_clean_up(df_poi)
        df_poi.head()
        db=connect_db() #establish connection
        Session = sessionmaker(bind=db)
        session = Session()
        save_to_database_ORM(session, df_poi[df_poi['poi_id']!=8])
        print(session.query(PointsOfInterest).count())

    else:
        print("The POPULATE_DB flag is false... exiting without action")
