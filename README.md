# Toronto Walks
## **Goal**

Generate walks around Toronto to suit a user's preferred starting point, amount of time and selected interests

## Problem Statement

One of my favourite pass times is to wander through a new neighbourhood or area of the city I haven't seen before and happen across interesting buildings / sites.  However, after a while, it can be hard to think of new places to explore or to plan a route that's likely to include something interesting.  This tool will attempt to address that issue by generating custom walk routes for users depending on

- where they want to start their walk
- how long they want to spend
- what their interests are (i.e. architecture (if so, what styles or architects), a particular historical period (i.e. 1850's Toronto), general interests etc)

## Overview

- TorontoWalks will gather points of interest from various open source listings of Toronto historical buildings, historical plaques, public art etc and store those in a database.  
- Will use GeoCoder api to apply lat/long coordinates to all POI (Points of Interest)



### Proposed Flow of Application

![TorontoWalks_Flow](https://github.com/ag2816/TorontoWalks/blob/master/docs/TorontoWalks_Flow.png)

## Data Sources

- **ACOToronto WebSite**: The ACOToronto website contains the TOBuilt database -- an open source database of images and information about buildings and structures in toronto.
  http://www.acotoronto.ca/tobuilt_new_detailed.php
- **Archidont Website**: Archindont is a database of architectural information and citations to periodical articles and books about buildings in Toronto.  http://archindont.torontopubliclibrary.ca

## Database Design

![ERD](https://github.com/ag2816/TorontoWalks/blob/master/docs/TorontoWalks_ERD.png)

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
