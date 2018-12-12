import pandas as pd
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values
import geopy.distance
import geocoder
import os
from dotenv import load_dotenv, find_dotenv
import numpy as np
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import googlemaps


# # Try with Google OR Tools

def get_google_key(get_map_obj= True):
    load_dotenv(find_dotenv())
    # load environment variables
    SECRET_KEY = os.getenv("GOOGLE_KEY")
    if get_map_obj == False:
        return SECRET_KEY

    gmaps = googlemaps.Client(key=SECRET_KEY)
    return gmaps


# Distance callback
def create_distance_callback(dist_matrix):
    # Create a callback to calculate distances between cities.

    def distance_callback(from_node, to_node):
        # return int(dist_matrix[from_node][to_node])
        #print(dist_matrix[from_node][to_node])
        return dist_matrix[from_node][to_node]

    return distance_callback


def make_dist_matrix(num_rows):
    #my_dist_matrix = np.zeros((len(walk_stops), len(walk_stops)))
    dist_matrix = np.zeros((num_rows,num_rows))
    return dist_matrix

def euclid_distance(x1, y1, x2, y2):
    # Euclidean distance between points.
    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return dist


def create_distance_matrix(locations):
    # Create the distance matrix.
    #gmaps = get_google_key()
    size = len(locations)
    dist_matrix = {}


    for from_node in locations.keys():
        dist_matrix[from_node] = {}
        for to_node in locations.keys():
            x1 = locations.get(from_node)[0]
            y1 = locations.get(from_node)[1]
            x2 = locations.get(to_node)[0]
            y2 = locations.get(to_node)[1]
            dist_matrix[from_node][to_node]  = geopy.distance.geodesic((x1,y1), (x2,y2)).meters
    return dist_matrix


#dist_matrix
#https://developers.google.com/optimization/routing/tsp
def create_routing_model(walk_stops, dist_matrix ):
    tsp_size = len(walk_stops)
    num_routes = 1
    depot = 0
    best_guess=[]

    # Create routing model
    if tsp_size > 0:
        routing = pywrapcp.RoutingModel(tsp_size, num_routes, depot)
        search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
        # Create the distance callback.
        dist_matrix=create_distance_matrix(walk_stops)
        dist_callback = create_distance_callback(dist_matrix)
        #print(dist_matrix)
        routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)
        # Solve the problem.
        assignment = routing.SolveWithParameters(search_parameters)
        if assignment:
            # Solution distance.
            #print ("Total distance: " + str(assignment.ObjectiveValue()) + " meters\n")
            # Display the solution.
            # Only one route here; otherwise iterate from 0 to routing.vehicles() - 1
            route_number = 0
            index = routing.Start(route_number) # Index of the variable for the starting node.
            route = ''
            while not routing.IsEnd(index):
                # Convert variable indices to node indices in the displayed route.
                print(routing.IndexToNode(index))
                best_guess.append(routing.IndexToNode(index))
                route += str(walk_stops[routing.IndexToNode(index)]) + ' -> '
                index = assignment.Value(routing.NextVar(index))
            route += str(walk_stops[routing.IndexToNode(index)])
            #print ("Route:\n\n" + route)
        else:
            print( 'No solution found.')
    else:
        print ('Specify an instance greater than 0.')
    return best_guess
