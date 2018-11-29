# Web Scraping Process and Challenges

## Site 1: ACOToronto

### Main Page

This website has a search form on the landing page that allows you to search by a number of criteria, including building style.  

![ACOToronto Search](https://github.com/ag2816/TorontoWalks/blob/master/docs/aco_toronto_search_page.png)

I scraped this form and called each style page in turn, using the syntax: http://www.acotoronto.ca/search_buildingsDB_d2.php?MainStyle={style} (i.e. http://www.acotoronto.ca/search_buildingsDB_d2.php?MainStyle=American%20colonial)

### Style Pages

I ran into a couple **challenges**

- the style page called with the "MainStyle" parameter redirects to a pid url, which has to be followed
- I eventually realized that each style page loaded at most 50 images at time and that you had to keep scrolling to get the rest of the records.  So I had to switch to using Selenium to grab those records

![Example end of records for Styles page](https://github.com/ag2816/TorontoWalks/blob/master/docs/aco_toronto_styles_page_end_of_records.png)



### Individual Building Pages

example url: http://www.acotoronto.ca/show_building.php?BuildingID=3883

![Example Building Page](https://github.com/ag2816/TorontoWalks/blob/master/docs/aco_toronto_building_page.png)

**Challenge**: The labels Name & Location, Status, Year Completed etc are stored in divs with class building_info, while the values immediately follow those labels in a div with class building_info2.  Unfortunately the value divs have no identifying information of their own

![example html](https://github.com/ag2816/TorontoWalks/blob/master/docs/aco_toront_building_page_html.png)

so I set up a dictionary with the labels of interest and looped through all the divs with class building_info.  When I found a match, I grabbed the text of the next tag

## Site 2: Archidont 

URL: <http://archindont.torontopubliclibrary.ca>

![achidont landing page](https://github.com/ag2816/TorontoWalks/blob/master/docs/Archidont_landing_page.png)

This site groups buildings by types which you can loop through alphabetically

![archidont buildings by type](https://github.com/ag2816/TorontoWalks/blob/master/docs/archidont_alphabetical_by_type_page.png)

When you click into a specific building type, the page lists details about all the buildings of that type and separates them with an image

![archidont building type page](https://github.com/ag2816/TorontoWalks/blob/master/docs/archidont_building_type_page.png)



* The building information is stored in alternating blockquote and center (dates and journals)

![archidont building info](https://github.com/ag2816/TorontoWalks/blob/master/docs/archidont_building_type_page_html.png)







## Site 3: Toronto Plaques

tbd

## Site 4: Cabbagetown People

tbd

# Other Issues

* after loading all entries into database, found 35 rows with invalid address to which GeoCoder couldn't assign a latitude / longitude
* ![rows with invalid addresses](https://github.com/ag2816/TorontoWalks/blob/master/docs/RowsWithInvalidAddresses.png)

* some addresses had a missing street number (i.e. 0 Yonge Street) while others were missing the city information (i.e. 7 Mossom Place)
* 
