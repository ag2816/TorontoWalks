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
- A blog entry on how I used SQLAlchemy ORM to set up the database can be found here: https://medium.com/dataexplorations/sqlalchemy-orm-a-more-pythonic-way-of-interacting-with-your-database-935b57fd2d4d



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

* need to measure similarity between user's stated preferences and each available POI to try to find the closest matches

#### Stage 4: Find best stops within range of Starting Coordinates

* 

#### Stage 5: Find Optimal Route

* 

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



## Project Deployment

- Created projects on Docker Images, hosted in Digital Ocean
  - Used Docker-Compose to get three docker instances to talk to each other
    - PostgreSQL
    - NGINX
    - Web (Flask web application)

## Project Organization

- used Trello to manage tasks
- used LucidChart to create flow chart of planned application flow and Database entity relationship diagram (ERD)