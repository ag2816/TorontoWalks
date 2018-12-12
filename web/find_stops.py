## COllection of methods to find stops for the walk
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

    types_weight_percent = 0.6
    prefs_weight_percent = 0.2
    ages_weight_percent = 0.2

    if sim_method =='cosine_similarity':
        # print('cosine_similarity')
        # avail_interests = list(df_features.columns)
        # types_list = [i for i, x in enumerate(avail_interests) if 'poi_type_' in x]
        # types_sim=cosine_similarity(df_features.iloc[:,types_list], df_user.iloc[:,types_list])
        #
        # prefs_list = [i for i, x in enumerate(avail_interests) if 'pref_' in x]
        # prefs_sim = cosine_similarity(df_features.iloc[:,prefs_list], df_user.iloc[:,prefs_list])
        #
        # ages_list = [i for i, x in enumerate(avail_interests) if 'build_century_' in x]
        # ages_sim = cosine_similarity(df_features.iloc[:,ages_list], df_user.iloc[:,ages_list])
        #
        # #print(types_sim, prefs_sim, ages_sim)
        #
        # cosine_sim=(types_sim * types_weight_percent) + (prefs_sim *prefs_weight_percent) + (ages_sim *ages_weight_percent)

        cosine_sim = cosine_similarity(df_features,df_user)
        user_matches = pd.DataFrame(cosine_sim, columns=['user_match']) # convert to df for ease
        df_poi['sim_rating'] = user_matches

       # # cosine_sim = cosine_similarity(df_features,df_user)
       #  user_matches = pd.DataFrame(cosine_sim, columns=['user_match']) # convert to df for ease
       #  user_matches.sort_values('user_match', ascending=False, inplace=True) # sort from best to worst matches
       #
       #  for ix,row in user_matches.iterrows():
       #      # now find matches close to target of interest - ix is row in dataframe that matches to user match
       #      df_poi.loc[ix,'sim_rating'] = row.user_match
    else:

        # update weights to put priority on choice of building / plaque / art
        num_features = len(list(df_features.columns))
        weight_of_importance=np.ones(num_features)
        weight_of_importance[-3:]= 10#.2

#         weight_of_importance[0:-3]=.4/(num_features-3)
#         weight_of_importance

        # spatial method requires numpy arrays
        np_features = df_features.values #as_matrix()
        U=df_user.values #as_matrix()

        site_prefs=[]
        for i in range(0,len(np_features)):
            np_features[i,:]
            if sim_method== 'spatial':
               # print('spatial')
                result = spatial.distance.cosine(np_features[i,:], U, weight_of_importance)
            else:
                #print('hamming')
                result = spatial.distance.hamming(np_features[i,:], U)
            res_dict = {'ix': i, 'dist': result}
            site_prefs.append(res_dict)
        # convert similarity rating for each POI to a df
        df_site_prefs = pd.DataFrame(site_prefs)
        df_site_prefs.sort_values('dist', ascending=False, inplace=True)

        for ix,row in df_site_prefs.iterrows():
            # now find matches close to target of interest
            df_poi.loc[ix,'sim_rating'] = row.dist


    df_poi.sort_values('sim_rating', inplace=True, ascending=False)
    return df_poi


#
# def find_similarity(df_features, df_user,  df_poi, sim_method=cosine_similarity):
#     '''
#     INPUTS: "vectorized" dataframes of POIS and User
#     RETURNS: dataframe of Points of Interest sorted from highest to lowest similarity rating to the user prefs
#     '''
#     df_poi['sim_rating'] = 0
#     if sim_method ==cosine_similarity:
#         cosine_sim = cosine_similarity(df_features,df_user)
#         user_matches = pd.DataFrame(cosine_sim, columns=['user_match']) # convert to df for ease
#         user_matches.sort_values('user_match', ascending=False, inplace=True) # sort from best to worst matches
#        # for ix,row in user_matches.iloc[0:20,:].iterrows():
#         for ix,row in user_matches.iterrows():
#             # now find matches close to target of interest - ix is row in dataframe that matches to user match
#             df_poi.loc[ix,'sim_rating'] = row.user_match
#     else:
#         # spatial method requires numpy arrays
#         np_features = df_features.as_matrix()
#         U=df_user.as_matrix()
#        # weight_of_importance=[0.05,0.05,0.05,0.05,.27,.26,.27]
#         site_prefs=[]
#         for i in range(0,len(np_features)):
#             np_features[i,:]
#             result = spatial.distance.cosine(np_features[i,:], U)#, weight_of_importance)
#             res_dict = {'ix': i, 'dist': result}
#             site_prefs.append(res_dict)
#         # convert similarity rating for each POI to a df
#         df_site_prefs = pd.DataFrame(site_prefs)
#         df_site_prefs.sort_values('dist', ascending=False, inplace=True)
#
#         for ix,row in df_site_prefs.iterrows():
#             # now find matches close to target of interest
#             df_poi.loc[ix,'sim_rating'] = row.dist
#
#     df_poi.sort_values('sim_rating', inplace=True, ascending=False)
#     return df_poi

def find_points_in_area(df, lat, long, num_points, max_distance, inflate_amount = 1.4):
    '''
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

    ## TODO: colllect all possible stops and then cluster?

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

# def find_points_in_area(df, lat, long, num_points, max_distance):
#     '''
#     Loop through POI dataframe that has already been sorted by similiarty to user prefs (from highest to lowest match)
#     find distance for each POI from starting point in meters
#     Check if is within the desired radius of the starting point to be a candidate stop
#     --> if yes, collect it in the avail_points list and increment the count of found stops
#     We have a few POIs with the exact same coordinates (sometimes due to duplicates, but also because some buildings may have multiple artworks)
#     --> this causes problems for google maps plotting of stops and also a fairly pointless walk with a bunch of stops in the same area
#     --> for now, only include the first POI at a given set of coordinates.  Use sets to make sure our coordinates are unique
#     due to some dirty data, have a few POIs with the same coordinates so skip repeats )
#     Once we have found the desired number of stops, exit
#
#     RETURNS: filtered dataframe of candidate stops with ~length of num_points
#
#     ## TODO: colllect all possible stops and then cluster?
#
#     '''
#
#     avail_points = []
#     found_points =0
#     prev_coord = (0,0) # sometimes have two points at same lat/long -- don't count as an extra stop
#     prev_coords = set({})
#     for ix, row in df.iterrows():
#
#         curr_pt = geopy.distance.geodesic((row['latitude'], row['longitude']), (lat, long)).meters
#
#         if curr_pt<= max_distance:
#             c=(row['latitude'], row['longitude'])
#             coord = set({c})
#             if prev_coords.issuperset(coord) == False:
#                 # only include POIs whose coordinates are not already in our list of stops
#                 prev_coords.add(c)
#                 my_dict={}
#                 my_dict =row.to_dict()
#                 my_dict['dist_start'] = curr_pt
#                 avail_points.append(my_dict)
#                 found_points +=1
#                 if found_points > num_points:
#                     break
#     df_2 = pd.DataFrame(avail_points)
#     return df_2

def cluster_stops(df, num_points):
    # get our unique lats and longs of the traps and cluster them
    points = df[['latitude', 'longitude']].drop_duplicates().values

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
    return df#, my_plot
