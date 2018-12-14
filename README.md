# Toronto Walks
## **Goal**

Generate walks around Toronto to suit a user's preferred starting point, amount of time and selected interests

## Problem Statement

One of my favourite pass times is to wander through a new neighbourhood or area of the city I haven't seen before and discover interesting buildings and sites.  However, after a while, it can be hard to think of new neighbourhoods to explore or to plan a route that's likely to include something I haven't seen before.  This tool will attempt to address that issue by generating custom walk routes for users depending on

- where they want to start their walk
- how much time they have
- their preferences and interests (i.e. architecture or public art, particular historical periods, subjects etc)

## Executive Summary

TorontoWalks is a walk generator that generates suggested walking routes based on user interests, starting point and desired walk duration.  The database of Toronto Points of Interest (POIs) was collected by web-scraping several websites (ACOToronto, TorontoPlaques, Archidont etc) and by using city of Toronto Open Data (public art works).  

The main steps involved in developing this tool were:

1) Gathering data

2) Preparing the data and user preferences

3) Measuring similarity between POIs and user interests

4) Finding "best matched" stops within reasonable distance of starting point

5) Plotting optimal route between stops (Travelling Salesman Problem)

The success of the generated walks is measured primarily by walk distance (is it achievable in time allotted?) and measure of similarity to user preferences.  I used cosine-similarity to find the best matches between a user’s stated interests and the features stored for each Point of Interest. I then found the most similar stops within a reasonable radius of the starting point and used HDBScan clustering to find a geographically constrained subset of stops. Finally I used Google OR Tools to plot the optimal route

The resulting tool was packaged in a flask application with a google maps interface for picking the starting point and plotting the generated route.  The application is currently hosted in 3 docker images on DigitalOcean.

## Overview of Application

**Landing Page:**

The user enters their interests (most influential is whether they are interested in buildings, historical plaques or public art).  They can also enter free text in the interests field -- this could be an architectural style, historical figure in Toronto history or a category of interest (i.e. sports, explorers etc)

![TorontoWalks_Flow](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/TorontoWalks_LandingPage.png)

**Example Generated Walk**

The generated walk is plotted on a Google Map with color coded markers according to the stop type (building, plaque or art work)

![TorontoWalks_Flow](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/TorontoWalks_GeneratedWalk.png)

**Example Stops**

Clicking on one of the markers opens a information pane with more information about that Point of Interest (depending on what is available for that POI).  This can include the address, architectural style, build year, a picture and a more detailed description.  

The following screenshots show an example information page for a building and for a Historical Plaque

![TorontoWalks_Flow](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/TorontoWalks_ExampleBuildingStop.png)

![TorontoWalks_Flow](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/TorontoWalks_ExamplePlaqueStop.png)

### Data Sources

- **ACOToronto WebSite**: The ACOToronto website contains the TOBuilt database -- an open source database of images and information about buildings and structures in toronto.
  http://www.acotoronto.ca/tobuilt_new_detailed.php
- **Archidont Website**: Archindont is a database of architectural information and citations to periodical articles and books about buildings in Toronto.  http://archindont.torontopubliclibrary.ca
- **Toronto Plaques**: http://torontoplaques.com/Menu_Subjects.html Details on heritage plaques in Toronto
- **Cabbagetown People**: http://www.cabbagetownpeople.ca/biographies/plaquees/ Information about plaques in Cabbagetown
- **Toronto Public Art**: City of Toronto Open Data https://www.toronto.ca/city-government/data-research-maps/open-data/open-data-catalogue/#86b0f307-216b-3fe1-4653-c32ed9b6bf5c

### Technologies Used

- Python
- Pandas
- pipenv - for local storage of credentials
- SQLAlchemy - interaction with database
- Beautiful Soup - Web Scraping
- Selenium - web scraping of scroll-to-load-more web pages
- Postgres database
- Docker for hosting database and flask web app
- Geocoder to get lats/longs of addresses
- SKLearn pipelines
- DataFrameMapper
- HDBSCANclustering
- Flask 
- Google Maps Api

## Project Notes

### Flow of Application

This flow chart outlines the major steps involved in generating a custom walk.  More detail on each of these steps is included below

![TorontoWalks_Flow](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/TorontoWalks_AppFlow.png)

### Preparation Stage: Gathering Data

* A more detailed overview of the websites scraped for this project, and what challenges I encountered, can be found here: https://github.com/ag2816/TorontoWalks/blob/master/docs/Web%20Scraping%20Process%20and%20Challenges.md
* Most of the data was gathered by web-scraping websites with good information about Toronto landmarks (listed above in Sources).  I used BeautifulSoup for most of it, but the ACOToronto.ca website had a continuous scroll-to-load-more functionality that required the use of Selenium.  In addition, information on public art works was loaded from Toronto Open Data
* Once the data was gathered, I pushed it through a cleanup pipeline to
  * remove duplicates
  * fix addresses (often incomplete, either with a missing city or Province or with an invalid street number (i.e. 0 Yonge Street, likely entered as a placeholder)).  When I first tried to get lats and longs, found that a number of sites in York had been geo-coded into New York State
  * Use GeoCoder to lookup latitude and longitude for each POI
  * engineer some new features
    * Build Decade
    * Simple POI type (building, plaque or art)
  * store in Database (PostgreSQL database hosted on Docker Image).  

![Data Gathering Flow](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/TorontoWalks_GatherData.png)

### Application Flow: User requests walk

#### Stage 1: Prepare Data

* DataFrameMapper --> label binarizes Simple POI Type and build Century
* Pipeline --> uses CountVectorizer to convert POI Architectural Style, Category and Name to vector with 0/1 for whether word is present or not. 
  * not every POI has a style or category
  * don't want to present user with a hundred different checkboxes for possible architectural styles and categories --> prefer to let them type in their interests and try to find a match based on that
  * Count Vectorizer -- removed digits, converted text to lowercase, and took top 5000 terms

#### Stage 2: Get user interests

* user fills out form 
* create a vector with same columns and dimensions as for POIs above (feed through same trained pipeline)

#### Stage 3: Measure Similarity

**Challenge**: need to measure similarity between user's stated preferences and each available POI to try to find the closest matches

##### Recommender Engines

This is not quite a recommender engine problem, but it's worth reviewing the basic concepts as they are related problems.

There are two main types of Recommender Engines:

- Collaborative
  - assume that similar people like similar things -- for example "people who buy x also buy y" (i.e. Amazon's Recommender engine)
- Content Based
  -  have matrix of features (i.e. keywords) about items and a vector of information about whether a user liked/disliked /ranked each item
  - Then find cosine similarity between the two vectors
  - Try to find items that are similar to those that a given user liked in the past

 **Implications for TorontoWalks**

The Walk Generator is somewhat similar to a content-based recommender engine, where we have a matrix of features (keywords) about each POI.  But we do NOT have access to user ratings of POIs.  Perhaps over time, we could develop that history by saving user generated walks and providing a way to like/dislike each stops.  But, for now, the application uses a simplified version of content filtering by generating a vector based on the user's stated preferences and comparing that to each POI using Cosine similarity.

##### Cosine Similarity

This is the formula for Cosine Similarity from Wikipedia

![](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/CosineSimilarityFormula.png)

The following graphic (from http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/) provides a good visualization of how this works.

![](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/CosineSimilarityDemo.png)

In the image below, the "sentences" represent "points of interest" and the formula is measuring similarity between terms like build century, poi type (art, building or plaque) and words used in the name, style or category)

![](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/CosineSimilarityVectorSpaceDemo.png)

##### Find Similarity Function

Similarity between user interests and POIs is calculated in the find_similarity() function in find_stops.py.  The application is currently using SKLearn cosine_similarity

**Input**

df_features: vectorized feature matrix of key terms for each poi

![](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/FindSimilarity_input_dffeatures.png)

df_user: matching vector of user preferences

![](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/FindSimilarity_input_dfusers.png)

**Calculations**

The first step is to calculate the cosine_similarity and then create a dataframe from it

```cosine_sim = cosine_similarity(df_features,df_user)
cosine_sim = cosine_similarity(df_features,df_user)
user_matches = pd.DataFrame(cosine_sim, columns=['user_match']) # convert to df for ease
```

Now each can store that as a new sim_rating column in our original dataframe

```
df_poi['sim_rating'] = user_matches
df_poi.sort_values('sim_rating', inplace=True, ascending=False)
```

**Outputs**

This method returns the original, full POI dataframe sorted based on the user's preference



![](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/FindSimilarity_output.png)

#### Stage 4: Find best stops within range of Starting Coordinates

* Now we need to trim this list back to reality and find the best matched stops within a reasonable distance of the walk starting point.
* The calculation is based on the assumptions that a user could comfortably visit 12 POIs in an hour within a radius of 1 KM (1000 meters) from the starting point
* More detailed information on this function can be found here: https://github.com/ag2816/TorontoWalks/blob/master/docs/FindPointsWithinDistance.md
* This worked well and returned decent walks.  But sometimes the stops ended up clustered on the extreme edges of the available radius around the starting point, making for a much longer walk than desired.
* The following image is a perfect example of a generated walk that could clearly have benefited from some clustering

![](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/ExampleWalkBeforeClustering.png)

* My solution was to add an extra step to cluster the found "best" stops in a more concentrated geographic cluster. 

**Clustering Overview**

* HDBSCAN is a density-based clustering algorithm, which means that it tries to find areas of the dataset that have a higher density of points than the rest of the dataset.  It is generally more interpretable than K-Means and has fewer hyperparameters to tune
* It supports a variety of distance metrics, including Haversine distance, which properly handles distance between lat/long coordinates (converted to radians)
* Tuned HDBScan so it produces 1 output cluster with the desired number of stops (all other stops are labelled as outliers)

* More detailed information on the actual clustering function used can be found here: https://github.com/ag2816/TorontoWalks/blob/master/docs/FindPointsWithinDistance.md

  ##### Impact of Adding Clustering

* Adding clustering made a statistically significant reduction in the total distance of the walks.  Before clustering, the average distance for a 1 hour walk (with distance of 1km/hr) was 4794.75 m, while after clustering, the average distance was 3,351m.   According to the T-Test, we would have only a .00014% chance of observing this large a difference in route distances if the two methods were equivalent

  ![](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/ClusteringImpact_ReducedWalkDistance.png)

  **Example walks:**

  * The following screenshots show examples of clustering applied to walks.  The green dots are the excluded stops, while the blue dots are the clustered stops included in the walk.  
  * Walk 1: starting King St / Simcoe for 2 hour walk

  ![](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/ClusteringImpact_Ex1.png)

  Walk 2: starting Avenue Road / Bloor for 2 hour walk



  ![](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/ClusteringImpact_Ex2.png)



#### Stage 5: Find Optimal Route

Now that we have a set of stops to include in our walk, we need to plot the best route between those stops.  This is an optimization problem and, specifically, a type of classic problem known as the Travelling Salesman Problem (TSP).  

##### Travelling Salesman Problem Overview

Here is a good overview of the problem from https://developers.google.com/optimization/routing/tsp:

> Back in the days when salesmen traveled door-to-door hawking vacuums and encyclopedias, they had to plan their routes, from house to house or city to city. The shorter the route, the better. Finding the shortest route that visits a set of locations is an exponentially difficult problem: finding the shortest path for 20 cities is much more than twice as hard as 10 cities.
>
> An exhaustive search of all possible paths would be guaranteed to find the shortest, but is computationally intractable for all but small sets of locations. For larger problems, optimization techniques are needed to intelligently search the solution space and find near-optimal solutions.

The following diagram (also from the above google website) shows an example of trying to find the path between nodes with the lowest sum of weights

![](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/tsp.svg)

##### Option 1 to address: Genetic Algorithm

The first tool I tried to use was a genetic algorithm.  I found a very clear example here, https://github.com/ZWMiller/PythonProjects/blob/master/genetic_algorithms/evolutionary_algorithm_traveling_salesman.ipynb (which I discovered through https://towardsdatascience.com/travel-time-optimization-with-machine-learning-and-genetic-algorithm-71b40a3a4c2), which I borrowed and slightly adapted.  

The idea is that you 

* generate a set of random guesses (possible order of stops)
* for each guess, calculate the travel time between the stops (using geopy.distance.geodesic)
* then find the best performing guess and select those to "breed" the next generation of guessses
  * Combine elements of two "parent" guesses to create a new child guess
  * include some amount of randomness for genetic diversity
* loop a set number of times
* track the best guess (lowest travel time)

This generated largely reasonable routes, but was very slow and not workable for a web application.  So I did some more research and came across Google OR Tools

##### Option 2: Google OR Tools

https://developers.google.com/optimization/routing/tsp

OR-Tools is an open source set of tools to solve optimization problems.  It uses combinatorial optimization to find the best solution to a problem out of a very large set of possible solutions.  

It was easy to implement and very fast at returning routes.  Generally all the generated routes have made sense

**Code**

- specify a solver

  - - tsp_size = # of stops
    - num_routes (1 for our case)
    - depot = start and end point (n/a for us)

- ```routing = pywrapcp.RoutingModel(tsp_size, num_routes, depot)
  routing = pywrapcp.RoutingModel(tsp_size, num_routes, depot)
  ```

- create a distance callback: The distance callback is a function that computes the distance between any two cities.

#### Stage 7: Add "Find best walk" feature

Once all the other pieces were working, I wanted to add a "find best walk" option that would try to find a walk that most closely matched your interests without being tied to a starting point.  

**Challenge**: Given that the "best matches" for a user's interests are spread out all over the city, how do I find a cluster of geographically close items?  Generally there won't be enough points in one area to make a complete walk, so need to flesh out with additional stops

**Approach**

* The function tries to find a cluster of relevant stops among the top 20 best matches (or fewer if the user's preference only matched with a very small subset of stops)
* extracts the lat / long of the first of those clustered stops and returns it as the starting point for a "regular" walk generation
* idea is that the find_points_in_area function should include the other stops in area since they're highly similar (attempting to simplify problem)

**Results**

![](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/ExampleWalk_SurpriseMe_BeauxArts_StartingPage.png)

![](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/ExampleWalk_SurpriseMe_BeauxArts_Results.png)

![](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/ExampleWalk_SurpriseMe_BeauxArts_SampleStop.png)

### Web Application Design

* used flask
* google maps api to plot coordinates
* Challenges
  * changing size and color of markers
  * centering label on marker
  * getting dynamic popups for each stop
* Bootstrap for form design

## Database Design

![ERD](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/TorontoWalks_ERD.png)

- Database: PostgreSQL hosted on Docker on Digital Ocean
- Use SQLAlchemy ORM to interact with database

### Testing

* Developed a function that generates walks based on a variety of test cases (walk starting points, durations and user preferences) and logs the criteria, walk generated, total distance of walk, average distance between stops, average similarity score etc
* Used this to compare performance before/after introduction changes to application (such as adding clustering)

## Project Deployment

- Created projects on Docker Images, hosted in Digital Ocean
  - Used Docker-Compose to get three docker instances to talk to each other
    - PostgreSQL
    - NGINX
    - Web (Flask web application)

from http://www.patricksoftwareblog.com/wp-content/uploads/2017/06/Docker-Application-Architecture-2.png:

![](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/Docker-Application-Architecture-2.png)

Details on Docker setup can be found here: https://github.com/ag2816/TorontoWalks/blob/master/docs/DockerSetup.md

## Project Organization

- used Trello to manage tasks
- used LucidChart to create flow chart of planned application flow and Database entity relationship diagram (ERD)

## Notes

- Project-related blog entries:
  - how I used SQLAlchemy ORM to set up the database can be found here: https://medium.com/dataexplorations/sqlalchemy-orm-a-more-pythonic-way-of-interacting-with-your-database-935b57fd2d4d
  - How to match up lat/long with specific neighbourhoods defined in a shape file: https://medium.com/dataexplorations/working-with-open-data-shape-files-using-geopandas-how-to-match-up-your-data-with-the-areas-9377471e49f2
  - How to create choropleth maps in Altair: https://medium.com/dataexplorations/creating-choropleth-maps-in-altair-eeb7085779a1