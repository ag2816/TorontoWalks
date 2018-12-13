
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
import re

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.impute import SimpleImputer
from sklearn_pandas import DataFrameMapper, CategoricalImputer
from sklearn.preprocessing import (
    StandardScaler, LabelBinarizer
)
#from genetic_algorithm import *
from optimization import *
from find_stops import *

from geopy.geocoders import Nominatim # convert an address into latitude and longitude values
import geopy.distance
import geocoder

DEBUG=0

import flask
from flask import  render_template, request, redirect
from flask import Markup
app = flask.Flask(__name__)

def make_session():
    db=connect_db() #establish connection / creates database on first run
    Session = sessionmaker(bind=db)
    session = Session()
    return db, session


import logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename='../logs/makerecommendations.log',level=logging.DEBUG)


def get_descriptors_for_poi(row):
    '''
    INPUT: accepts 1 row of a dataframe /series
    RETURNS: string
    '''
    descriptor = ""
    keys = ['style', 'category','name']
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


def text_preprocess(s):
    '''
    removes digits from string,
    converts to lowercase
    '''
    s =  re.sub('\d+', '', s).lower()

    return(s)

# pipeline used for converting descriptors to word vector
# remove numbers, return lower case
pipe = Pipeline([
    ('cv', CountVectorizer(preprocessor = text_preprocess,
        ngram_range=(1,1),
        lowercase=True,
        max_features=5000,
        strip_accents='unicode'))
])

def get_user_profile(df_cols, user_sel_prefs, prefs, pipe):
    '''
    INPUTS
    df_cols: list with cols of "vectorized" dataframe
    user_sel_prefs: list with selected cols that interest user (poi types and eras)
    prefs: string with keywords describing user's iterests
    pipeline: for vectorizing prefs
    OUTPUTS
    vector with 0/1 for presence /absense of each feature in user prefs
    '''

    user_prefs = np.zeros(len(df_cols))

    for pref in user_sel_prefs:
        indx=df_cols.index(pref)
        user_prefs[indx] = 1

    df_user=pd.DataFrame(user_prefs).T
    #columns are numerically named-- change to match original df
    col_size=list(range(len(df_cols)))
    df_user.rename(columns = dict(zip(col_size, df_cols )), inplace=True)

    d2=transform_pipeline_for_column(pd.Series(prefs), pipe)

    df_user=pd.concat([df_user,d2], axis=1)
    return df_user


def create_walking_route(starting_lat, starting_long, duration, prefs, user_interests, starting_pt_pref):
    #determine distance, num pos to visit based on duration
    pts_per_hour = 12
    max_dist_per_hour = 1000 # meters
    num_points = duration * pts_per_hour
    max_distance=duration * max_dist_per_hour
    logging.debug(f"NEW CALL for user with interests {user_interests}, starting point {starting_lat},{starting_long} and duration {duration}")

    #set up our data
    # pipe = pickle.load(open("pipe.pkl","rb"))
    # df_features = pickle.load(open("df_features.pkl","rb"))
    # df_poi = pickle.load(open("df_poi.pkl","rb"))
    # avail_interests= pickle.load(open("avail_interests.pkl","rb"))
    df_poi = get_pois_as_df() # this stores the full info about each POI (we need to produce meaningful Stop info)
    df_features=df_poi.copy() # this will store our vectorized interests
    poi_mapper.fit(df_features)
    df_features= poi_mapper.transform(df_features)

    avail_interests = list(df_features.columns)
    avail_interests.remove('descriptors')

    # vectorize descriptors (text vectorizer based on Category, Architectural Style and stop name)
    pipe_desc = pipe.fit(df_features['descriptors'])
    df_trans = transform_pipeline_for_column(df_features['descriptors'], pipe_desc)
    # merge back into dataframe produced by dataframe mapper (POITypes and Build Century)
    df_features.drop(columns='descriptors', inplace=True)
    df_features = pd.concat([ df_trans, df_features], axis=1)
    # sort to make sure column order is consistent with user prefs (gets mixed up during concat operation above)
    df_features=df_features.reindex(sorted(df_features.columns), axis=1)

    # create user interest vector using trained pipeline
    df_user = get_user_profile(avail_interests, user_interests, prefs, pipe)

    # calculate similiarity between user prfs and POIs
    df_poi=find_similarity(df_features, df_user, df_poi)

    # if user wants "best match" walking
    if starting_pt_pref== 'starting_pt_random':
        starting_lat, starting_long = find_walk_starting_pt_by_interest(df_poi)
    #starting_lat, starting_long

    # narrow down to subset of stops in vicitinity of starting point
    df_filtered = find_points_in_area(df_poi, starting_lat, starting_long, num_points, max_distance)

     # cluster stops to get better fit
    df_filtered = cluster_stops(df_filtered, num_points)
    df_filtered=df_filtered.reset_index()

    # find optimal route
    guess, df_filtered, walk_stops=find_optimal_route(df_filtered, starting_lat, starting_long, method='google')
    df_filtered.sort_values("order", inplace=True, ascending=True)

    stops_ordered = []
    for stop in guess:
        stops_ordered.append(walk_stops[stop])
    stops_ordered2=json.dumps(stops_ordered)

    return  df_filtered, stops_ordered2, guess,starting_lat, starting_long

@app.route("/")
def hello():
    google_key = get_google_key( False   )
    return render_template('toronto_walks.html', google_key = google_key,)


@app.route('/page')
def page():
    google_key = get_google_key( False   )
    return render_template('toronto_walks.html', google_key = google_key,)


def create_display_text(row):
    color_dict = {'Art': 'bg-primary', 'Building': 'bg-info', 'Plaque': 'bg-secondary'}

    detail_html = f'<div class="card"><div class="card-header {color_dict[row["poi_type_simple"]]}"><h6>{row["name"]}</h6></div><div class="card-body">'
    detail_html += f'<p class="card-text">Stop Type: {row["poi_type_simple"]}</p>'
    detail_html += f'<p class="card-text">Address: {row["address"]}</p>'
    detail_html += f'<p class="card-text">Build Year: {row["build_year"]}</p>'
    if row["style"] != None:
        detail_html += f'<p class="card-text">Style: {row["style"]}</p>'
    if row["external_url"] != None:
        detail_html +=f'<p>Link for More Information: <a href="{row["external_url"]}" target="_blank">{row["external_url"]}</a></p>'
    if row["image_url"] != None:
        detail_html +=f'<p><img src={row["image_url"]} width="200" heigh="200"><p>{row["details"]}</p>'
    detail_html += f'<p class="card-text">Source: {row["source"]}</p>'
    detail_html +=' </div></div>'

    return  detail_html


@app.route('/result', methods=['POST', 'GET'])
def result():
    '''Gets prediction using the HTML form'''
    if flask.request.method == 'POST':

        # read inputs from form
        inputs = flask.request.form
        starting_lat = float(inputs['latitude'])
        starting_long= float(inputs['longitude'])
        duration = float(inputs['duration'])
        prefs = inputs['pref']
        # checkboxes -- the list is the id of the checkboxes that are CHECKED
        # unchecked boxes do not appear here
        user_interests = request.form.getlist('user_interests')
        starting_pt_pref = request.form['starting_pt_radio']
        # if starting_pt_random, then want "best fit" walk with no fixed starting points

        # invoke the logic
        df_filtered, stops_ordered2, guess,starting_lat, starting_long=create_walking_route(starting_lat, starting_long, duration, prefs, user_interests, starting_pt_pref)

        # prep for display
        google_key = get_google_key( False   )

        # html for details box for each poi
        stop_text=[ create_display_text(row) for ix, row in df_filtered.iterrows() ]
        stop_text = Markup(stop_text)

        # defined the coloured icons used to identify the different stop types
        color_dict = {'Art': 'http://maps.google.com/mapfiles/ms/micons/blue.png', 'Building': 'http://maps.google.com/mapfiles/ms/micons/lightblue.png', 'Plaque': 'http://maps.google.com/mapfiles/ms/micons/pink.png'}
        #color_dict = {'Art': 'http://maps.google.com/mapfiles/kml/paddle/blu-blank.png', 'Building': 'http://maps.google.com/mapfiles/kml/paddle/grn-blank.png', 'Plaque': 'http://maps.google.com/mapfiles/kml/paddle/pink-blank.png'}
        icons=[ color_dict[row] for row in df_filtered['poi_type_simple'].values ]
        icons = Markup(icons)

        return render_template('map_route.html', route=guess, starting_lat =starting_lat, starting_long =starting_long, ordered_stops = stops_ordered2, google_key = google_key, stop_text=stop_text, stop_icons=icons)


if __name__ == '__main__':
    '''Connects to the server'''
    app.run()
