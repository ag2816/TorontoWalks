# Toronto Walks
## **Goal**

Generate walks around Toronto to suit a user's preferred starting point, amount of time and selected interests

## Problem Statement

One of my favourite pass times is to wander through a new neighbourhood or area of the city I haven't seen before and discover interesting buildings and sites.  However, after a while, it can be hard to think of new places to explore or to plan a route that's likely to include something interesting.  This tool will attempt to address that issue by generating custom walk routes for users depending on

- where they want to start their walk
- how much time they have
- their preferences and interests (i.e. architecture or public art, particular historical periods, subjects etc)

## Executive Summary

1. An executive summary:

- What is your goal?
- Where did you get your data?
- What are your metrics?
- What were your findings?
- What risks/limitations/assumptions affect these findings?
- What risks/limitations/assumptions affect these findings?

## Overview

- TorontoWalks will gather points of interest from various open source listings of Toronto historical buildings, historical plaques, public art etc and store those in a database.  
- Will use GeoCoder api to apply lat/long coordinates to all POI (Points of Interest)
- Project-related blog entries:
  - how I used SQLAlchemy ORM to set up the database can be found here: https://medium.com/dataexplorations/sqlalchemy-orm-a-more-pythonic-way-of-interacting-with-your-database-935b57fd2d4d
  - How to match up lat/long with specific neighbourhoods defined in a shape file: https://medium.com/dataexplorations/working-with-open-data-shape-files-using-geopandas-how-to-match-up-your-data-with-the-areas-9377471e49f2
  - How to create choropleth maps in Altair: https://medium.com/dataexplorations/creating-choropleth-maps-in-altair-eeb7085779a1



### Proposed Flow of Application

![TorontoWalks_Flow](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/TorontoWalks_Flow.png)

## Data Sources

- **ACOToronto WebSite**: The ACOToronto website contains the TOBuilt database -- an open source database of images and information about buildings and structures in toronto.
  http://www.acotoronto.ca/tobuilt_new_detailed.php
- **Archidont Website**: Archindont is a database of architectural information and citations to periodical articles and books about buildings in Toronto.  http://archindont.torontopubliclibrary.ca
- **Toronto Plaques**: http://torontoplaques.com/Menu_Subjects.html Details on heritage plaques in Toronto
- **Cabbagetown People**: http://www.cabbagetownpeople.ca/biographies/plaquees/ Information about plaques in Cabbagetown
- **Toronto Public Art**: City of Toronto Open Data https://www.toronto.ca/city-government/data-research-maps/open-data/open-data-catalogue/#86b0f307-216b-3fe1-4653-c32ed9b6bf5c

## Database Design

![ERD](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/TorontoWalks_ERD.png)

- Database: PostgreSQL hosted on Docker on Digital Ocean
- Use SQLAlchemy ORM to interact with database

## Metric of Success

tbd

## Technologies Used

- Python
- Pandas
- pipenv - for local storage of credentials
- SQLAlchemy - interaction with database
- Beautiful Soup - Web Scraping
- Selenium - web scraping of scroll-to-load-more web pages
- Postgres database
- Docker for hosting postgres
- Flask (coming...)
- Geocoder to get lats/longs of addresses
- Trello - task Management
- LucidChart - Flow Chart and Database ERD design
- Google Maps Api
- SKLearn pipelines
- HDBSCANclustering

## Project Notes
### Preparation Stage: Gathering Data

* A more detailed overview of the websites scraped for this project, and what challenges I encountered, can be found here: https://github.com/ag2816/TorontoWalks/blob/master/docs/Web%20Scraping%20Process%20and%20Challenges.md
* Feature Engineering
  * Build Century / Build Decade
  * Simple POI Type (art, plaque, building)
* Data Cleaning
  * remove duplicates
  * fix addresses (often incomplete, either with a missing city or Province or with an invalid street number (i.e. 0 Yonge Street, likely entered as a placeholder)).  When I first tried to get lats adn longs, found that a number of sites in York had been geo-coded into New York State
  * Use GeoCoder to lookup latitude and longitude for each POI
  * store in Database
* Used PostgreSQL database hosted in Docker image
* Used SQLAlchemy ORM to interact with Database (please refer to my blog post on setting this up https://medium.com/dataexplorations/sqlalchemy-orm-a-more-pythonic-way-of-interacting-with-your-database-935b57fd2d4d)



### Application Flow: User requests walk

#### Stage 1: Prepare Data

* DataFrameMapper --> label binarizes Simple POI Type and build Century
* Pipeline --> uses CountVectorizer to convert POI Architectural Style, Category and Name to vector with 0/1 for whether word is present or not. 
  * not every POI has a style or category
  * don't want to present user with a hundred different checkboxes for possible architectural styles and categories --> prefer to let them type in their interests and try to find a match based on that

#### Stage 2: Get user prefs

* user fills out form 
* create a vector with same columns and dimensions as for POIs above (feed through same trained pipeline)

#### Stage 3: Measure Similarity

Challenge: need to measure similarity between user's stated preferences and each available POI to try to find the closest matches

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

The Walk Generator is somewhat similar to a content-based recommender engine, where we have a matrix of features (keywords) about each POI.  But we do NOT have access to user ratings of POIs.  Perhaps over time, we could develop that history by saving user generated walks and providing a way to reject or rate the stops/walk.  But, for now, the application uses a simplified version of content filtering by generating a vector based on the user's stated preferences and comparing that to each POI using Cosine similarity.

##### Cosine Similarity

Will use Cosine Similarity to do this

Available libraries

- sklearn cosine_similarity
- spatial.distance.cosine
- spatial.distance.hamming

This is the formula for Cosine Similarity from Wikipedia

![](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/CosineSimilarityFormula.png)

The following graphic (from http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/) provides a good visualization of how this works.

![](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/CosineSimilarityDemo.png)

The image below, the "sentences" are "points of interest" and the formula is measuring similarity between terms like build century, poi type (art, building or plaque) and words used in the name, style or category)

![](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/CosineSimilarityVectorSpaceDemo.png)



##### Find Similarity Function

This is calculated in the find_similarity() function in find_stops.py.  This method accepts a sim_method argument to indicate which of the above tools should be used to calculate the similarity.  The application is currently using SKLearn cosine_similarity

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

Now each can add that as sim_rating column to our original dataframe

```
df_poi['sim_rating'] = user_matches
df_poi.sort_values('sim_rating', inplace=True, ascending=False)
```

**Outputs**

This method returns the original, full POI dataframe sorted based on the user's preference



![](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/FindSimilarity_output.png)

#### Stage 4: Find best stops within range of Starting Coordinates

* Now we need to trim this list back to reality and find the best matched stops within a reasonable distance of the walk starting point.

* Assumed user could generally comfortably visit 12 POIs in an hour within a radius of 1 KM (1000 meters) from the starting point

* But found that the generated walks were often unreasonably long and so decided to add an extra step to cluster the found "best" stops in a more concentrated geographic cluster.

  * used HDBSCAN which can use geographic (haversine) distance rather than simple Euclidean disance ("flat earth").  Tuned it so it produces 1 output cluster with the desired number of stops (all other stops are labelled as outliers)

* More information on these functions can be found here: https://github.com/ag2816/TorontoWalks/blob/master/docs/FindPointsWithinDistance.md

  ##### Impact of Adding Clustering

* Adding clustering made a statistically significant reduction in the total distance of the walks.  Before clustering, the average distance for a 1 hour walk (with distance of 1km/hr) was 4794.75 m, while after clustering, the average distance was 3,351m.   According to the T-Test, we would have only a .00014% chance of observing this large a difference in route distances if there were no difference

  ![](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/ImpactofClusteringonWalkDistance.png)

  **Example walks:**

  * green dots are the excluded stops
  * Walk 1: starting King St / Simcoe for 2 hour walk

  ![](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/ClusteringImpact_Ex1.png)

  Walk 2: starting Avenue Road / Bloor for 2 hour walk



  ![](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/ClusteringImpact_Ex2.png)

#### Stage 5: Find Optimal Route

Now that we have a set of stops to include in our walk, we need to plot the best route between those stops.  This is an optimization problem and, specifically, a type of classic problem known as the Travelling Salesman Problem (TSP).  

##### Travelling Salesman Problem Overview



##### Option 1 to address: Genetic Algorithm

The first tool I tried to use was a genetic algorithm



This generated largely reasonable routes, but was VERY slow and not workable for a web application.  So I did some more research and came across Google OR Tools

##### Option 2: Google OR Tools

https://developers.google.com/optimization/routing/tsp



#### Stage 6: Display walk to user

* 

#### Stage 7: add Find best walk feature

"find best walk"

Called when user doesn't specify starting point.  Tries to find a cluster of relevant stops based on user interest and build a walk from there
​    To simplify the problem, tries to find a cluster of relevant stops among the top 20 best matches
​    Then extracts the lat / long of the first of those clustered stops and returns it.  idea is that the find points in area function should include the other stops in area since they're highly similar
​    (attempting to simplify problem)



### Web Application Design

* used flask
* google maps api to plot coordinates
* Challenges
  * changing size and color of markers
  * centering label on marker
  * getting dynamic popups for each stop
* Bootstrap for form design



### Testing

* test method 



## Project Deployment

- Created projects on Docker Images, hosted in Digital Ocean
  - Used Docker-Compose to get three docker instances to talk to each other
    - PostgreSQL
    - NGINX
    - Web (Flask web application)

from http://www.patricksoftwareblog.com/wp-content/uploads/2017/06/Docker-Application-Architecture-2.png:

![](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/Docker-Application-Architecture-2.png)

## Project Organization

- used Trello to manage tasks
- used LucidChart to create flow chart of planned application flow and Database entity relationship diagram (ERD)