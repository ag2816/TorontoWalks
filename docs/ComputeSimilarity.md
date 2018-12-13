# Compute Similarity

Need to measure similarity between Vector of user preferences and Matrix of Point of Interest features.  

## Recommender Engines

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

## Cosine Similarity

Will use Cosine Similarity to do this

Available libraries

* sklearn cosine_similarity
* spatial.distance.cosine
* spatial.distance.hamming

This is the formula for Cosine Similarity from Wikipedia

![](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/CosineSimilarityFormula.png)

The following graphic (from http://blog.christianperone.com/2013/09/machine-learning-cosine-similarity-for-vector-space-models-part-iii/) provides a good visualization of how this works.

![](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/CosineSimilarityDemo.png)

The image below, the "sentences" are "points of interest" and the formula is measuring similarity between terms like build century, poi type (art, building or plaque) and words used in the name, style or category)

![](https://github.com/ag2816/TorontoWalks/blob/master/docs/images/CosineSimilarityVectorSpaceDemo.png)



## Find Similarity Function

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


