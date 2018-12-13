# Find points within distance

 Calculated by find_points_in_area in find_stops.py

**Inputs**

- POI Dataframe: that has already been sorted by similarity to user prefs
- Start latitude / longitude
- Num_Points: number of stops to include in walk (based on walk duration)
- max_distance: maximum distance to look for stops around starting point
- inflate_amount: when adding step to cluster found stops (to get more concentrated points) need to grab more than minimum "num points" so we have something to cluster.  This is the "markup" to apply.  Defaults to 1.4 (so find 40% more potential stops).
  - trade-off: as we grab more spots, risk diluting match to user prefs.  but the fewer we grab, the more spread out they may be around the 1km/hr radius

**Calculations**

Loops through POI dataframe (already sorted from best to worst match) and for each POI:

-  find distance for each POI from starting point in meters

- ```Get 
  curr_pt = geopy.distance.geodesic((row['latitude'], row['longitude']), (lat, long)).meters
  ```

- Check if is within the desired radius of the starting point to be a candidate stop

  - If Yes:
    - validate that another stop with the exact same coordinates isn't already in the list.  If it is, then skip this stop for now (otherwise, we have a problem plotting the stops in Google Maps and the point of a walk is to travel distance) .  See below for an example (this is mainly an issue due to multiple artworks found in same complex, but can also be an issue with dirty/duplicate data)
    - if Ok, collect the stop in the avail_points list and increment the count of found stops
  - Once we have found the desired number of stops, exit

**Outputs**

  filtered data frame of candidate stops with ~length of num_points

```find
avail_points = []
    found_points =0
    prev_coord = (0,0) # sometimes have two points at same lat/long -- don't count as an extra stop
    prev_coords = set({})
    buffered_num_pts = num_points * inflate_amount

    for ix, row in df.iterrows():

        curr_pt = geopy.distance.geodesic((row['latitude'], row['longitude']), (lat, long)).meters

        if curr_pt<= max_distance:
            c=(row['latitude'], row['longitude'])
            coord = set({c})
            if prev_coords.issuperset(coord) == False:
                # only include POIs whose coordinates are not already in our list of stops
                prev_coords.add(c)
                my_dict={}
                my_dict =row.to_dict()
                my_dict['dist_start'] = curr_pt
                avail_points.append(my_dict)
                found_points +=1
                if found_points > buffered_num_pts:
                    break
    df_2 = pd.DataFrame(avail_points)
    return df_2
```

**Challenges**

Example problem with multiple stops at the same coordinates

![1544458393916](https://github.com/ag2816/TorontoWalks/blob/master/docs/find_similarity_duplicateCoords.png)

This results in an mess on Google Maps where you can't access the extra stops

![1544458490147](https://github.com/ag2816/TorontoWalks/blob/master/docs/find_similarity_duplicatecoords_plottingissue.png)