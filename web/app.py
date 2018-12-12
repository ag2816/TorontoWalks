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

#from genetic_algorithm import *
from optimization import *
from find_stops import *

from geopy.geocoders import Nominatim # convert an address into latitude and longitude values
import geopy.distance
import geocoder


DEBUG=0

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn_pandas import DataFrameMapper, CategoricalImputer
from sklearn.preprocessing import (
    StandardScaler, LabelBinarizer, Imputer, FunctionTransformer,PolynomialFeatures, OrdinalEncoder
)


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


def add_features(df):
    df['build_century'] = df['build_year_clean'].apply(lambda x: x//100*100)
    return df

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


poi_mapper = DataFrameMapper([
    ('build_century',[CategoricalImputer(replacement="n/a"), LabelBinarizer()]),
     ('poi_type_simple',[CategoricalImputer(replacement="n/a"), LabelBinarizer()]),
], df_out=True)


def get_user_profile(df_cols, user_sel_prefs):
    '''
    df_cols: list with cols of "vectorized" dataframe
    user_sel_prefs: list with selected cols that interest user
    '''

    user_prefs = np.zeros(len(df_cols))

    for pref in user_sel_prefs:
        indx=df_cols.index(pref)
        user_prefs[indx] = 1

    df_user=pd.DataFrame(user_prefs).T
    return df_user

def create_walking_route(starting_lat, starting_long, duration, prefs, user_interests):
    #determine distance, num pos to visit based on duration
    pts_per_hour = 12
    max_dist_per_hour = 1000 # meters
    num_points = duration * pts_per_hour
    max_distance=duration * max_dist_per_hour
    logging.debug(f"NEW CALL for user with interests {user_interests}, starting point {starting_lat},{starting_long} and duration {duration}")

    #set up our data
    df_poi = get_pois_as_df()
    df_features=df_poi.copy()
    poi_mapper.fit(df_features)
    df_features= poi_mapper.transform(df_features)

    avail_interests = list(df_features.columns)
    df_user = get_user_profile(avail_interests, user_interests)

    df_poi=find_similarity(df_features, df_user, df_poi)
    df_filtered = find_points_in_area(df_poi, starting_lat, starting_long, num_points, max_distance)
     # add clustering
    df_filtered = cluster_stops(df_filtered, num_points)
    df_filtered=df_filtered.reset_index()

    # find optimal route
    guess, df_filtered, walk_stops=find_optimal_route(df_filtered, starting_lat, starting_long, method='google')
    df_filtered.sort_values("order", inplace=True, ascending=True)

    stops_ordered = []
    for stop in guess:
        stops_ordered.append(walk_stops[stop])
    stops_ordered2=json.dumps(stops_ordered)

    return  df_filtered, stops_ordered2, guess

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

        # invoke the logic
        df_filtered, stops_ordered2, guess=create_walking_route(starting_lat, starting_long, duration, prefs, user_interests)

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
