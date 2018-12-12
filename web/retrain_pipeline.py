## This code recreates and saves the dataframe of POIs from teh database, as well as the vectorizef
## this is intended to reduce DB load as the stops are not changeing frequently

from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv, find_dotenv
from models import connect_db, PointsOfInterest, ArchitecturalStyles, Architects,POICategories

import math
from copy import copy
import pandas as pd
import os
import time
import numpy as np
import json
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.impute import SimpleImputer
from sklearn_pandas import DataFrameMapper, CategoricalImputer
from sklearn.preprocessing import (
    StandardScaler, LabelBinarizer
)


def make_session():
    db=connect_db() #establish connection / creates database on first run
    Session = sessionmaker(bind=db)
    session = Session()
    return db, session

def get_descriptors_for_poi(row):
    '''
    INPUT: accepts 1 row of a dataframe /series
    RETURNS: string
    '''
    descriptor = ""
    keys = ['style', 'category']
    for key in keys:
        if row[key]!=None:
            descriptor += row[key]
    return descriptor

def add_features(df):
    df['build_century'] = df['build_year_clean'].apply(lambda x: x//100*100)
    df['descriptors'] = df.apply(lambda row:get_descriptors_for_poi(row), axis=1 )
    return df

def transform_pipeline_for_column(my_col, pipe):
    '''
    apply pipeline to selected column in the dataframe
    '''
    returned_words = pipe.transform(my_col)
    tmp_df = pd.DataFrame(returned_words.toarray(), columns=pipe.named_steps['cv'].get_feature_names())
    ## add a prefix
    return tmp_df.add_prefix('pref_')

def get_pois_as_df():

    db, session = make_session()
    sql='''SELECT poi.*, styl.style, pcat.category
    FROM points_of_interest poi
    LEFT JOIN architectural_styles styl on (styl.poi_id = poi.poi_id)
    LEFT JOIN poi_categories pcat on (pcat.poi_id = poi.poi_id)
    order by poi.poi_id
    '''
    df = pd.read_sql_query(sql, db)
    df= add_features(df)
    session.close()
    return df

# converts POI dataframe to vector for similarity comparisons
poi_mapper = DataFrameMapper([
    ('build_century',[CategoricalImputer(replacement="n/a"), LabelBinarizer()]),
    ('poi_type_simple',[CategoricalImputer(replacement="n/a"), LabelBinarizer()]),
    ('descriptors',[CategoricalImputer(replacement="n/a")]),
], df_out=True)

# pipeline used for converting descriptors to word vector
pipe = Pipeline([
    ('cv', CountVectorizer(ngram_range=(1,1),
            lowercase=True,
            max_features=5000,
            strip_accents='unicode'))
])

df_poi = get_pois_as_df()
df_features=df_poi.copy()
poi_mapper.fit(df_features)
df_features= poi_mapper.transform(df_features)

avail_interests = list(df_features.columns)
avail_interests.remove('descriptors')

# vectorize descriptors
pipe_desc = pipe.fit(df_features['descriptors'])
df_trans = transform_pipeline_for_column(df_features['descriptors'], pipe_desc)
df_features.drop(columns='descriptors', inplace=True)
df_features = pd.concat([ df_trans, df_features], axis=1)
# order to match user prefs!
df_features=df_features.reindex(sorted(df_features.columns), axis=1)

# pickle
pickle.dump(pipe, open("pipe.pkl", "wb"))
pickle.dump(df_features, open("df_features.pkl", "wb"))
pickle.dump(df_poi, open("df_poi.pkl", "wb"))
pickle.dump(avail_interests, open("avail_interests.pkl", "wb"))
