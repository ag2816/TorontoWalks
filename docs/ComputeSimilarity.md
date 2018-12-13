# Compute Similarity

Need to measure similarity between Vector of user preferences and Matrix of Point of Interest features.  Will use Cosine Similarity to do this

Available libraries

* sklearn cosine_similarity
* spatial.distance.cosine
* spatial.distance.hamming

## Cosine similarity

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


