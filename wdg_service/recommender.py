# Import libraries
from collections import Counter, OrderedDict
from itertools import chain
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from keras import models
import warnings
warnings.filterwarnings("ignore")

def get_df():
    """ Returns main dataframe used in the project """

    # Path to file
    hotels_path = "../where_do_we_go_from_here/data/clean_hotels_test.csv"
    # Dataframe
    hotels_df = pd.read_csv(hotels_path, usecols = ["city", "country", "hotel_name", "rating", 
                                                    "address", "popularity_rating", "locality", "price",
                                                "landmark", "URL"])
    return hotels_df

def get_baseline_df():
    """ Return dataframe with baseline recommendations """
    # Path to file
    recommendations_path = "../where_do_we_go_from_here/data/baseline_recommendations.csv"

    # Dataframe
    recommendations_df = pd.read_csv(recommendations_path, usecols = ["city", "country", "hotel_name", "rating", 
                                                    "address", "popularity_rating", "locality", "price",
                                                "landmark", "URL"])
    return recommendations_df

def get_model():
    """ Return model architecture and weights """

    # Import embeddings model and weights
    model = models.load_model("../where_do_we_go_from_here/models/embeddings_fourth_attempt.h5")
    return model.load_weights("../where_do_we_go_from_here/models/embeddings_fourth_attempt_weights.h5")

def baseline():
    """ Returns a json object with baseline recommendations to address the cold-start problem """
    recommendations_df = get_baseline_df()
    recommendations = recommendations_df.to_dict(orient = "list")
    return recommendations

def get_int_mapping(dataframe, column):
    """ Returns index, reverse_index, and list of unique items in a pandas datframe """

    # Convert series to list
    column_to_list = dataframe[column].tolist()

    # Find set of unique items and convert to a list
    unique_items_list = list(set(column_to_list))

    # Create indexes for each item
    item_index = {item: idx for idx, item in enumerate(unique_items_list)}
    index_item = {idx: item for item, idx in item_index.items()}

    return item_index, index_item, unique_items_list

def get_embeddings(layer_name):
    """ Given a model and a layer name, this function returns the 
    normalized embedding [weights] for said layer """

    # Get model
    model = get_model()

    # Get layer
    item_layer = model.get_layer(layer_name)

    # Get weights
    item_weights = item_layer.get_weights()[0]

    # Normalize the embeddings so that we can calculate cosine similarity
    item_weights = item_weights / np.linalg.norm(item_weights, axis = 1).reshape((-1, 1))

    return item_weights

def find_similar(name, weights, index_name = "hotel_name", n = 20, 
                        filtering = False, filter_name = None):
    """ Return json object with most similar items """

    hotels_df = get_df()

    # Mapping of items to integers with get_int_mapping().
    city_index, index_city, unique_cities = get_int_mapping(hotels_df, "city")
    country_index, index_country, unique_countries = get_int_mapping(hotels_df, "country")
    hotel_index, index_hotel, unique_hotels = get_int_mapping(hotels_df, "hotel_name")
    rating_index, index_rating, unique_ratings = get_int_mapping(hotels_df, "rating")
    popularity_index, index_popularity, unique_popularities = get_int_mapping(hotels_df, "popularity_rating")
    locality_index, index_locality, unique_localities = get_int_mapping(hotels_df, "locality")
    price_index, index_price, unique_prices = get_int_mapping(hotels_df, "price")
    landmark_index, index_landmark, unique_landmarks = get_int_mapping(hotels_df, "landmark")
   
    # Select index and reverse index
    if index_name == "city":
        index = city_index
        rindex = index_city
    if index_name == "country":
        index = country_index
        rindex = index_country
    if index_name == "hotel_name":
        index = hotel_index
        rindex = index_hotel
    if index_name == "rating":
        index = rating_index
        rindex = index_rating
    if index_name == "popularity_rating":
        index = popularity_index
        rindex = index_popularity
    if index_name == "locality":
        index = locality_index
        rindex = index_locality
    if index_name == "price":
        index = price_index
        rindex = index_price
    if index_name == "landmark":
        index = landmark_index
        rindex = index_landmark
    
    # Check name is in index
    try:
        # Calculate dot product between item/property and all others
        distances = np.dot(weights, weights[index[name]])
    except KeyError:
        print(" {} Not Found.".format(name))
        return
    
    # Sort distances from smallest to largest
    sorted_distances = np.argsort(distances)
        
    # Find the most similar
    closest = sorted_distances[-n:]
    
    # Limit results by filtering
    filter_ = None
    filtered_results = []
    results_dict = {}
    if filtering:
        for idxs, rows in hotels_df.iterrows():
            if hotels_df.at[idxs, index_name] == name:
                filter_ = hotels_df.at[idxs, filter_name]
                break
        match_df = hotels_df[hotels_df[filter_name].str.match(filter_)]
        match_df = match_df.reset_index(drop = True)
        match_df["distance"] = None
        for idxs, rows in match_df.iterrows():
            item = match_df.at[idxs, index_name]
            distance = np.dot(weights[index[item]], weights[index[name]])
            match_df.loc[match_df.index[idxs], "distance"] = distance
        match_df = match_df.sort_values(by = ["distance"], axis = 0, ascending = False)
        results_dict = match_df.to_dict(orient = "list")
        list_of_filtered_results = match_df[index_name].to_list()
        for item in list_of_filtered_results:
            if item not in filtered_results:
                filtered_results.append(item)   
        return results_dict
    
    # Print the most similar item and distances
    items = {"city": [], "country": [], "hotel": [], "url": [], "landmark": [], "locality": [], "rating": []}
    for c in reversed(closest):
        for idxs, rows in hotels_df.iterrows():
            if hotels_df.at[idxs, index_name] == rindex[c]:
                items["city"].append(hotels_df.at[idxs, "city"])
                items["country"].append(hotels_df.at[idxs, "country"])
                items["hotel"].append(hotels_df.at[idxs, "hotel_name"])
                items["url"].append(hotels_df.at[idxs, "URL"])
                items["landmark"].append(hotels_df.at[idxs, "landmark"])
                items["locality"].append(hotels_df.at[idxs, "locality"])
                items["rating"].append(hotels_df.at[idxs, "rating"])
    return items

              