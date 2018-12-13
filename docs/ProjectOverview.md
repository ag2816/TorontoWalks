# **TorontoWalks Capstone Project**

Amy Gordon

Goal

Generate walks around Toronto to suit a user's preferred starting point, amount of time and selected interests

# Executive Summary

1. An executive summary:

- What is your goal?
- Where did you get your data?
- What are your metrics?
- What were your findings?
- What risks/limitations/assumptions affect these findings?
- What risks/limitations/assumptions affect these findings?

## Problem Statement

One of my favourite pass times is to wander through a new neighbourhood or area of the city I haven't seen before and discover interesting buildings and sites.  However, after a while, it can be hard to think of new places to explore or to plan a route that's likely to include something interesting.  This tool will attempt to address that issue by generating custom walk routes for users depending on

- where they want to start their walk
- how much time they have
- their preferences and interests (i.e. architecture or public art, particular historical periods, subjects etc)

## Overview

- TorontoWalks will gather points of interest from various open source listings of Toronto historical buildings, historical plaques, public art etc and store those in a database.  
- Will use GeoCoder api to apply lat/long coordinates to all POI (Points of Interest)



### Proposed Flow of Application

![](C:\Users\blahjays\Documents\GitHubCode\Personal_Public\BuildingStyleClassifier\docs\TorontoWalks_Flow.png)

## Data Sources

- **ACOToronto WebSite**: The ACOToronto website contains the TOBuilt database -- an open source database of images and information about buildings and structures in toronto.
  http://www.acotoronto.ca/tobuilt_new_detailed.php
- **Archidont Website**: Archindont is a database of architectural information and citations to periodical articles and books about buildings in Toronto.  http://archindont.torontopubliclibrary.ca
- **Toronto Plaques**: Information on historical plaques located in Toronto http://torontoplaques.com/Menu_Subjects.html
- **Cabbagetown People**: Information on Cabbagetown Association plaques installed around Cabbagetown https://github.com/ag2816/TorontoWalks/blob/master/docs/
- **Toronto Public Art**: City of Toronto Open Data https://www.toronto.ca/city-government/data-research-maps/open-data/open-data-catalogue/#86b0f307-216b-3fe1-4653-c32ed9b6bf5c

## Database Design

![ERD](C:\Users\blahjays\Documents\GitHubCode\Personal_Public\BuildingStyleClassifier\docs\TorontoWalks_ERD.png)

* Database: PostgreSQL hosted on Docker on Digital Ocean
* Use SQLAlchemy ORM to interact with database

## Metric of Success

tbd

## Technologies Used

- General
  - Python
  - Pandas
  - 
- pipenv - for local storage of credentials
- SQLAlchemy - interaction with database
- Beautiful Soup - Web Scraping
- Selenium - web scraping of scroll-to-load-more web pages
- Postgres database
- Docker 
- Flask 
- Geocoder to get lats/longs of addresses
- Trello - task Management
- LucidChart - Flow Chart and Database ERD design
- Google Maps Api
- SKLearn pipelines
- HDBSCANclustering



"find best walk"

Called when user doesn't specify starting point.  Tries to find a cluster of relevant stops based on user interest and build a walk from there
​    To simplify the problem, tries to find a cluster of relevant stops among the top 20 best matches
​    Then extracts the lat / long of the first of those clustered stops and returns it.  idea is that the find points in area function should include the other stops in area since they're highly similar
​    (attempting to simplify problem)