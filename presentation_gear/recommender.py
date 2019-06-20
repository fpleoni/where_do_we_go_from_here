# Import libraries
from collections import Counter, OrderedDict
from itertools import chain
from  more_itertools import unique_everseen
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from keras import models
import warnings
import functools
import operator
warnings.filterwarnings("ignore")

def get_df():
    """ Returns main dataframe used in the project """

    # Path to file
    hotels_path = "../data/clean_hotels_scraped_v2.csv"
    # Dataframe
    hotels_df = pd.read_csv(hotels_path, usecols = ["city", "country", "hotel_name", "rating", 
                                                    "address", "popularity_rating", "locality", "price",
                                                "landmark", "URL"])
    return hotels_df

def get_model():
    """ Return model architecture and weights """

    # Import embeddings model and weights
    model = models.load_model("../models/nn_scraped_hotels.h5")
    model.load_weights("../models/nn_scraped_hotels_weights.h5")
    return model

def get_int_mapping(dataframe, column):
    """ Returns index, reverse_index, and list of unique items in a pandas datframe """

    # Convert series to list
    column_to_list = dataframe[column].tolist()

    # Find set of unique items and convert to a list
    unique_items_list = list(unique_everseen(column_to_list))

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

def find_similar(name, weights, index_name = "hotel_name", n = 10, plot = True, filtering = False, filter_name = None):
    """ Return most similar items """

    index = hotel_index
    rindex = index_hotel
    
    # Select index and reverse index
    if index_name == "city":
        index = city_index
        rindex = index_city
    if index_name == "country":
        index = country_index
        rindex = index_country
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
    hotel_name = []
    city = []
    country = []
    url = []
    landmark = []
    locality = []
    rating = []

    # Limit results by filtering
    filter_ = None
    filtered_results = []
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
        list_of_filtered_items = match_df[index_name].to_list()
        list_of_filtered_distances = match_df["distance"].to_list()
        list_of_filtered_results = list(zip(list_of_filtered_items, list_of_filtered_distances))
        for item in list_of_filtered_results[1:]:
            if item not in filtered_results:
                filtered_results.append(item)     
        if plot:
            # Find closest and most far away item
            closest = filtered_results[:n // 2]
            far_away = filtered_results[-n-1: len(filtered_results) - 1]
            to_plot = [c[0] for c in closest]
            to_plot.extend(c[0] for c in far_away)

            # Find distances 
            dist = [c[1] for c in closest]
            dist.extend(c[1] for c in far_away)  

            # Colors
            colors = ["darkturquoise" for _ in range(n)]
            colors.extend("hotpink" for _ in range(n // 2))

            # Data in DataFrame
            data = pd.DataFrame({"distance": dist}, index = to_plot)

            # Bar chart
            data["distance"].plot.barh(color = colors, figsize = (10, 8), edgecolor = "k", linewidth = 2)
            plt.xlabel("Cosine Similarity");
            plt.axvline(x = 0, color = "k");

            # Title
            name_str = "Most and Least Similar to {}".format(name)
            plt.title(name_str, x = 0.2, size = 28, y = 1.05)
            return None
        
        return None

    # Plot results
    if plot:
        # Find closest and most far away item
        far_away = sorted_distances[:n // 2]
        closest = sorted_distances[-n-1: len(distances) - 1]
        to_plot = [rindex[c] for c in far_away]
        to_plot.extend(rindex[c] for c in closest)
        
        # Find distances 
        dist = [distances[c] for c in far_away]
        dist.extend(distances[c] for c in closest)
        
        # Colors
        colors = ["hotpink" for _ in range(n // 2)]
        colors.extend("darkturquoise" for _ in range(n))
        
        # Data in DataFrame
        data = pd.DataFrame({"distance": dist}, index = to_plot)
        
        # Bar chart
        data["distance"].plot.barh(color = colors, figsize = (10, 8), edgecolor = "k", linewidth = 2)
        plt.xlabel("Cosine Similarity");
        plt.axvline(x = 0, color = "k");
        
        # Title
        name_str = "Most and Least Similar to {}".format(name)
        plt.title(name_str, x = 0.2, size = 28, y = 1.05)
        
        return None