# Methods around Optimisation of the route between stops
# the Travelling Salesman problems
import pandas as pd
import numpy as np
from optimization_genetic_algorithm import *
from optimization_ortools import *

def get_dict_of_stop_coords(df,starting_lat, starting_long, key_to_use='poi_id'):
    '''
    create dictionary of stops with lat/long coords
    first stop is the starting point
    '''
   # print(f'starting point ({starting_lat}, {starting_long})')
    walk_stops = {}
    walk_stops[0] = (starting_lat, starting_long)
    for ix,row in df.iterrows():

        if key_to_use=='poi_id':
            walk_stops[row['poi_id']] = (row['latitude'], row['longitude'])
        else:
            walk_stops[ix+1] = (row['latitude'], row['longitude'])

    return walk_stops


def add_stop_order_to_df(df, order_guess, key_to_use='poi_id'):
    df['order'] =0
    cnt = 1
    for ix in order_guess[1:]:
        if key_to_use=='poi_id':
            index = df[df_filtered['poi_id'] ==ix].index
        else:
            index = ix-1
        df.iloc[index,-1]=cnt
        cnt +=1
    return df


def find_optimal_route(df, starting_lat, starting_long, method='genetic'):
    '''
    Possible methods: genetic or google optimisation tools
    '''
    walk_stops = get_dict_of_stop_coords(df, starting_lat, starting_long)
    if method == 'genetic':
        current_generation = create_generation(list(walk_stops.keys()),population=500)
        fitness_tracking, best_guess = evolve_to_solve(current_generation, 100, 150, 70, 0.5, 3, 5, walk_stops, verbose=True)
        plot_guess(walk_stops, best_guess)
        # add order to df.
        df=add_stop_order_to_df(df, best_guess)
        order_guess = best_guess
    else:
        #method='google'
        walk_stops = get_dict_of_stop_coords(df, starting_lat, starting_long, key_to_use='count')
        dist_matrix=make_dist_matrix(len(walk_stops))
        order_guess=create_routing_model(walk_stops,dist_matrix)

        df=add_stop_order_to_df(df, order_guess,key_to_use='count')
    return order_guess, df, walk_stops
