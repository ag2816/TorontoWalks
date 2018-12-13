# This code uses a Genetic Algorithm to try to solve the Travelling Salesman problem
# Code was borrowed and slightly adapted from https://github.com/ZWMiller/PythonProjects/blob/master/genetic_algorithms/evolutionary_algorithm_traveling_salesman.ipynb

import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values
import geopy.distance
import geocoder


def create_guess(walk_stops):
    """
    Creates a possible path between all cities, returning to the original.
    Point 0 is our starting point and must stay as our starting point
    Input: List of City IDs
    """
    guess = copy(walk_stops)
   # print(f"before shuffle {guess}")
    start = guess.pop(0)# save our starting point and remove from dict
    np.random.shuffle(guess)
   # print(f"after shuffle {guess}")
   # guess[0] = start
    guess.insert(0,start)
    #guess.append(guess[0])
    return list(guess)

def create_generation(points, population=100):
    """
    Makes a list of guessed point orders given a list of point IDs.
    Input:
    points: list of point ids
    population: how many guesses to make
    """
    generation = [create_guess(points) for _ in range(population)]
    return generation


def travel_time_between_points(point_1, point_2):
    '''
    pass in coords for 2 points
    calculate distance between them in meters and then estimate walking time
    '''
    # typical walkign speed is 1.4m/sec
    speed = 1.4
    #find dist between 2 points
    dist = geopy.distance.geodesic(point_1, point_2).meters
    # return guess speed in seconds
    return dist/speed


def fitness_score(guess, walk_stops):
    """
    Loops through the points in the guesses order and calculates
    how much distance the path would take to complete a loop.
    Lower is better.
    """
    score = 0
    for ix, point_id in enumerate(guess[:-1]):
        score += travel_time_between_points(walk_stops[point_id], walk_stops[guess[ix+1]])
    return score

def check_fitness(guesses, walk_stops):
    """
    Goes through every guess and calculates the fitness score.
    Returns a list of tuples: (guess, fitness_score)
    """
    fitness_indicator = []
    for guess in guesses:
        fitness_indicator.append((guess, fitness_score(guess, walk_stops)))
    return fitness_indicator


def get_breeders_from_generation(guesses, walk_stops, take_best_N=10, take_random_N=5, verbose=False, mutation_rate=0.1):
    """
    This sets up the breeding group for the next generation. You have
    to be very careful how many breeders you take, otherwise your
    population can explode. These two, plus the "number of children per couple"
    in the make_children function must be tuned to avoid exponential growth or decline!
    """
    # First, get the top guesses from last time
    fit_scores = check_fitness(guesses, walk_stops)
    sorted_guesses = sorted(fit_scores, key=lambda x: x[1]) # sorts so lowest is first, which we want
    new_generation = [x[0] for x in sorted_guesses[:take_best_N]] # takes top 5
    best_guess = new_generation[0] # best guess is the best score

    if verbose:
        # If we want to see what the best current guess is!
        print(best_guess)

    # Second, get some random ones for genetic diversity
    for _ in range(take_random_N):
        ix = np.random.randint(len(guesses))
        new_generation.append(guesses[ix])

    # No mutations here since the order really matters.
    # If we wanted to, we could add a "swapping" mutation,
    # but in practice it doesn't seem to be necessary

    np.random.shuffle(new_generation)
    return new_generation, best_guess

def make_child(parent1, parent2):
    """
    Take some values from parent 1 and hold them in place, then merge in values
    from parent2, filling in from left to right with cities that aren't already in
    the child.
    """
    list_of_ids_for_parent1 = list(np.random.choice(parent1, replace=False, size=len(parent1)//2))
   # print(list_of_ids_for_parent1)
    child = [-99 for _ in parent1] # fill with placeholders so now all -99 values can be replaced with genes from parent 2

    for ix in range(0, len(list_of_ids_for_parent1)):
        child[ix] = parent1[ix]

    for ix, gene in enumerate(child):

        if gene == -99:
            for gene2 in parent2:
                if gene2 not in child:
                    child[ix] = gene2
                    break
    #child[-1] = child[0]
    return child

def make_children(old_generation, children_per_couple=1):
    """
    Pairs parents together, and makes children for each pair.
    If there are an odd number of parent possibilities, one
    will be left out.

    Pairing happens by pairing the first and last entries.
    Then the second and second from last, and so on.
    """
  #  print(old_generation)
    mid_point = len(old_generation)//2
#    print(mid_point)
    next_generation = []

    for ix, parent in enumerate(old_generation[:mid_point]):
        for _ in range(children_per_couple):
            next_generation.append(make_child(parent, old_generation[-ix-1]))
   # print(next_generation)
    return next_generation


def evolve_to_solve(current_generation, max_generations, take_best_N, take_random_N,
                    mutation_rate, children_per_couple, print_every_n_generations, walk_stops, verbose=False):
    """
    Takes in a generation of guesses then evolves them over time using our breeding rules.
    Continue this for "max_generations" times.
    Inputs:
    current_generation: The first generation of guesses
    max_generations: how many generations to complete
    take_best_N: how many of the top performers get selected to breed
    take_random_N: how many random guesses get brought in to keep genetic diversity
    mutation_rate: How often to mutate (currently unused)
    children_per_couple: how many children per breeding pair
    print_every_n_geneartions: how often to print in verbose mode
    verbose: Show printouts of progress
    Returns:
    fitness_tracking: a list of the fitness score at each generations
    best_guess: the best_guess at the end of evolution
    """
    fitness_tracking = []
    for i in range(max_generations):
        if verbose and not i % print_every_n_generations and i > 0:
            print("Generation %i: "%i, end='')
            print(len(current_generation))
            print("Current Best Score: ", fitness_tracking[-1])
            is_verbose = True
        else:
            is_verbose = False
        breeders, best_guess = get_breeders_from_generation(current_generation, walk_stops,
                                                            take_best_N=take_best_N, take_random_N=take_random_N,
                                                            verbose=is_verbose, mutation_rate=mutation_rate)
        fitness_tracking.append(fitness_score(best_guess, walk_stops))
        current_generation = make_children(breeders, children_per_couple=children_per_couple)

    return fitness_tracking, best_guess
