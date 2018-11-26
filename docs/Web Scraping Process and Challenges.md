# Web Scraping Process and Challenges

## Site 1: ACOToronto

### Main Page

This website has a search form on the landing page that allows you to search by a number of criteria, including building style.  

![ACOToronto Search](C:\Users\blahjays\Documents\GitHubCode\Personal_Public\BuildingStyleClassifier\docs\aco_toronto_search_page.png)

I scraped this form and called each style page in turn, using the syntax: http://www.acotoronto.ca/search_buildingsDB_d2.php?MainStyle={style} (i.e. http://www.acotoronto.ca/search_buildingsDB_d2.php?MainStyle=American%20colonial)

### Style Pages

I ran into a couple issues

- the style page called with the "MainStyle" parameter redirects to a pid url, which has to be followed
- I eventually realized that each style page loaded at most 50 images at time and that you had to keep scrolling to get the rest of the records.  So I had to switch to using Selenium to grab those records

![Example end of records for Styles page](C:\Users\blahjays\Documents\GitHubCode\Personal_Public\BuildingStyleClassifier\docs\aco_toronto_styles_page_end_of_records.png)



### Individual Building Pages

example url: http://www.acotoronto.ca/show_building.php?BuildingID=3883

![Example Building Page](C:\Users\blahjays\Documents\GitHubCode\Personal_Public\BuildingStyleClassifier\docs\aco_toronto_building_page.png)

The labels Name & Location, Status, Year Completed etc are stored in divs with class X, while the values immediately follow those labels in a div with class Y.  Unfortunately the value divs have no identifying information of their own

![example html](C:\Users\blahjays\Documents\GitHubCode\Personal_Public\BuildingStyleClassifier\docs\aco_toront_building_page_html.png)

hh