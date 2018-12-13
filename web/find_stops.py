## Collection of methods to find stops for the walk


import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values
import geopy.distance
import geocoder
from hdbscan import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial

def find_similarity(df_features, df_user,  df_poi, sim_method='cosine_similarity'):
    '''
    INPUTS: "vectorized" dataframes of POIS and User
    sim_method:
        * cosine_similarity=
        * spatial =
        * hamming
    RETURNS: dataframe of Points of Interest sorted from highest to lowest similarity rating to the user prefs
    '''
    df_poi['sim_rating'] = 0

    if sim_method =='cosine_similarity':

        cosine_sim = cosine_similarity(df_features,df_user)
        user_matches = pd.DataFrame(cosine_sim, columns=['user_match']) # convert to df for ease
        df_poi['sim_rating'] = user_matches

    else:

        # update weights to put priority on choice of building / plaque / art
        num_features = len(list(df_features.columns))

        # spatial method requires numpy arrays
        np_features = df_features.values #as_matrix()
        U=df_user.values #as_matrix()

        site_prefs=[]
        for i in range(0,len(np_features)):
            np_features[i,:]
            if sim_method== 'spatial':
               # print('spatial')
                result = spatial.distance.cosine(np_features[i,:], U)#, weight_of_importance)
            else:
                #print('hamming')
                result = spatial.distance.hamming(np_features[i,:], U)
            res_dict = {'ix': i, 'dist': result}
            site_prefs.append(res_dict)
        # convert similarity rating for each POI to a df
        df_site_prefs = pd.DataFrame(site_prefs)
        df_poi['sim_rating'] = df_site_prefs

    df_poi.sort_values('sim_rating', inplace=True, ascending=False)
    return df_poi


def find_points_in_area(df, lat, long, num_points, max_distance, inflate_amount = 1.4):
    '''
    INPUTS:
    df: list of all pois sorted from best to worst match
    lat/lng: starting latitude and longitude to search form
    num_points: desired number of stops to include in walk
    max_distance: max distance from starting point to look for stops
    inflate_amount: as part of clustering approach, need to grab mroe than minimum number of stops to have a better chance of getting a good cluster.  THis is the "markup" to apply
    OVERVIEW:
    Option 2: find a "buffer" of points beyond the specified desired number of stops and then use clustering on the returned matches
     to reduce the list back down to num_points
    Loop through POI dataframe that has already been sorted by similiarty to user prefs (from highest to lowest match)
    find distance for each POI from starting point in meters
    Check if is within the desired radius of the starting point to be a candidate stop
    --> if yes, collect it in the avail_points list and increment the count of found stops
    We have a few POIs with the exact same coordinates (sometimes due to duplicates, but also because some buildings may have multiple artworks)
    --> this causes problems for google maps plotting of stops and also a fairly pointless walk with a bunch of stops in the same area
    --> for now, only include the first POI at a given set of coordinates.  Use sets to make sure our coordinates are unique
    due to some dirty data, have a few POIs with the same coordinates so skip repeats )
    Once we have found the desired number of stops, exit

    RETURNS: filtered dataframe of candidate stops with ~length of num_points
    '''

    avail_points = []
    found_points =0
    prev_coord = (0,0) # sometimes have two points at same lat/long -- don't count as an extra stop
    prev_coords = set({})
    buffered_num_pts = num_points * inflate_amount

    for ix, row in df.iterrows():

        curr_pt = geopy.distance.geodesic((row['latitude'], row['longitude']), (lat, long)).meters

        if curr_pt<= max_distance:
            c=(row['latitude'], row['longitude'])
            coord = set({c})
            if prev_coords.issuperset(coord) == False:
                # only include POIs whose coordinates are not already in our list of stops
                prev_coords.add(c)
                my_dict={}
                my_dict =row.to_dict()
                my_dict['dist_start'] = curr_pt
                avail_points.append(my_dict)
                found_points +=1
                if found_points > buffered_num_pts:
                    break
    df_2 = pd.DataFrame(avail_points)
    return df_2


def cluster_stops(df, num_points):
    '''
    INPUTS:
    df: dataframe with limited number of candidate stops (likely 20-40% more than desired number of stops)
    num_points: desired number of stops in cluster
    OVERVIEW:
    Finds geographically close stops from available candidates
    Uses Haversine distance, based on radians
    Trying to get a single cluster with the exact number of desired stops.  So turn on allow_single_cluster (otherwise, HDBSCAN will not return just 1 group)
    RETURNS
    filtered dataframe that only contains stops included in the cluster
    '''
    # get our unique lats and longs of the pois and cluster them
    points = df[['latitude', 'longitude']].values #.drop_duplicates().values

    # convert to radians for HDBScan (per docs)
    rads = np.radians(points)

    clusterer = HDBSCAN(min_cluster_size=int(num_points), metric='haversine',allow_single_cluster=True)
    cluster_labels = clusterer.fit_predict(points)

    # add label column to df
    df['labels']=cluster_labels

    # drop points not included in cluster
    df = df[df['labels']==0]

    # drop labels column
    df.drop(columns='labels', inplace=True)
    return df

def find_walk_starting_pt_by_interest(df_poi, num_matches=20):
    '''
    INPUTS:
    df_poi: dataframe with POIs sorted from best match to worst
    num_matches: number of best matches to cluster against (the larger this is, the more likely the walk will be built on less relavant stops.  but the smaller it is, the mmore likely we won't be able to find a cluster of geographically close relevant stops)
    OVERVIEW
    Called when user doesn't specify starting point.  Tries to find a cluster of relevant stops based on user interest and build a walk from there
    To simplify the problem, tries to find a cluster of relevant stops among the top 20 best matches
    Then extracts the lat / long of the first of those clustered stops and returns it.  idea is that the find points in area function should include the other stops in area since they're highly similar
    (attempting to simplify problem)
    RETURNS:
    new starting lat and long for a walk.  If we can't find a match, default to Yonge/Dundas square

    '''
    starting_lat =43.656333
    starting_long=-79.380942
    # min points in clusterer
    min_cluster_size=6

    try:
        # if only a few points match the user interests, then we want to focus on them
        num_sim = len(df_poi[df_poi['sim_rating']>0])
        if num_sim < num_matches and num_sim > 0:
            num_matches=num_sim
            min_cluster_size=2

        # take the top twenty closest matches
        df=df_poi.iloc[0:num_matches,:].reindex()
        df_f=cluster_stops(df, min_cluster_size)
        if len(df_f) > 0:
            # extract lat and long of first point
            starting_lat=df_f.iloc[0,7]
            starting_long=df_f.iloc[0,8]
    except:
        pass # for now, we'll fall back to yonge/dundas square

    return starting_lat, starting_long
