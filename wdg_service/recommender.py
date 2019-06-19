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
    hotels_path = "../data/clean_hotels_test.csv"
    # Dataframe
    hotels_df = pd.read_csv(hotels_path, usecols = ["city", "country", "hotel_name", "rating", 
                                                    "address", "popularity_rating", "locality", "price",
                                                "landmark", "URL"])
    return hotels_df

def get_baseline_df():
    """ Return dataframe with baseline recommendations """
    
    # Path to file
    recommendations_path = "../data/baseline_recommendations.csv"

    # Dataframe
    recommendations_df = pd.read_csv(recommendations_path, usecols = ["city", "country", "hotel_name", "rating", 
                                                    "address", "popularity_rating", "locality", "price",
                                                "landmark", "URL"])
    return recommendations_df

def get_model():
    """ Return model architecture and weights """

    # Import embeddings model and weights
    model = models.load_model("../models/embeddings_fourth_attempt.h5")
    model.load_weights("../models/embeddings_fourth_attempt_weights.h5")
    return model

# Unpersonalized recommendations
def baseline():
    """ Returns a json object with baseline recommendations to address the cold-start problem """

    # Get list of most popular cities
    cities = ['London', 'Paris', 'Istanbul', 'New York', 'Rio de Janeiro', 'Amsterdam', 'Rome', 'Cancun', 
              'Tokyo', 'Berlin', 'Dubai', 'Madrid', 'Las Vegas', 'Barcelona', 'Prague', 'São Paulo', 'Natal', 
              'Florianópolis', 'Buenos Aires', 'Orlando', 'Bangkok', 'Kuala Lumpur', 'Miami Beach', 'Vienna', 
              'Porto Seguro', 'Milan', 'Foz do Iguaçu', 'Playa del Carmen', 'Osaka', 'Hamburg', 'Sydney', 
              'Armação dos Búzios','Melbourne', 'Maceió', 'Fortaleza', 'Porto de Galinhas', 'Acapulco', 'Lisbon', 
              'Singapore', 'Salvador Bahia','Puerto Vallarta', 'Gramado', 'Hong Kong', 
              'Malacca', 'Mexico City', 'Guarujá', 'Ubatuba', 'Munich', 'Venice', 
              'Budapest', 'Cartagena', 'Moscow', 'Edinburgh', 'Marrakech', 'Kyoto', 'Port Dickson', 
              'Campos do Jordão','Florence', 'Balneário Camboriú', 'Paraty', 'Antalya', 'Mazatlán', 
              'Caldas Novas', 'San Francisco', 'Dublin','Copenhagen', 'Cabo Frio', 'Mumbai', 
              'Cologne', 'Ankara', 'Guadalajara', 'Ilhabela', 'Seoul', 'Arraial do Cabo', 
              'Udaipur', 'Seville', 'Taipei City', 'St Petersburg', 'Miami', 'Québec City', 
              'Morro de São Paulo', 'Curitiba','Athens', 'João Pessoa', 'São Sebastião', 
              'Pattaya', 'Los Angeles', 'Bombinhas', 'Urayasu', 'Manchester','Praia da Pipa', 
              'Brussels', 'Brisbane', 'Lake Buena Vista', 'Stockholm', 'Frankfurt', 'Auckland', 
              'Anaheim', 'Fukuoka', 'Fort Lauderdale']
    
    indexes = []
    
    # Best reviews
    best_reviews = hotels_df[hotels_df["rating"] >= 4.5]
    best_reviews = best_reviews[best_reviews["popularity_rating"] > 100]
    
    # Sort best reviews by popularity rating and price
    best_reviews = best_reviews.sort_values(by = ["popularity_rating"], axis = 0, 
                         ascending = False)
    # Reset index
    best_reviews = best_reviews.reset_index(drop = True)
    
    for city in cities:
        # Get the hotels with 5.0 ratings that are at each of the top cities
        idxs = best_reviews[best_reviews["city"] == city].index
        indexes.append(idxs)
        
    flatten = functools.reduce(operator.iconcat, indexes, [])
    # Create lists to then create json object
    name = []
    city = []
    country = []
    url = []
    landmark = []
    locality = []
    rating = []
    popularity = []
    # Loop through all idxs in the flattened id list, and add the info at each idx row to list
    for i, idx in enumerate(flatten[:31]):
        city.append(best_reviews.at[idx, "city"])
        country.append(best_reviews.at[idx, "country"])
        name.append(best_reviews.at[idx, "hotel_name"])
        rating.append(best_reviews.at[idx, "rating"])
        popularity.append(best_reviews.at[idx, "popularity_rating"])
        landmark.append(best_reviews.at[idx, "landmark"])
        locality.append(best_reviews.at[idx, "locality"])
        url.append(best_reviews.at[idx, "URL"])
    # Create json object
    hotels = [{"name": n, "city": c + ", " + p, "url": u, "landmark": l,
                    "locality": t, "rating": r} for n, c, p, u, l, t, r 
                  in zip(name, city, country, url, landmark, locality, rating)]
    return hotels
    
# Unpersonalized recommendations
def citybased_recommendation_baseline(location):
    """ Returns top hotel recommendations for a """
    
    hotels_df = get_df()

    best_reviews = hotels_df[hotels_df["rating"] >= 4.0]
    best_reviews = best_reviews.sort_values(by = ["popularity_rating", "rating", "price"], axis = 0, 
                         ascending = [False, False, True])
    best_reviews = best_reviews.reset_index(drop = True)
    
    # Separate location in city and country
    location = location.split(",")
    city = location[0].strip()
    country = location[1]
    
    # Get the hotels that satisfy the keyword
    subset_of_df = best_reviews_2[best_reviews_2["city"] == city]
    idxs = list(subset_of_df[subset_of_df["country"] == country].index)
    
    # Create lists to store results
    name = []
    city = []
    country = []
    url = []
    landmark = []
    locality = []
    rating = []
    popularity = []
    
    # Loop through all idxs in the flattened id list, and add the info at each idx row to dictionary
    for idx in idxs:
        city.append(best_reviews.at[idx, "city"])
        country.append(best_reviews.at[idx, "country"])
        name.append(best_reviews.at[idx, "hotel_name"])
        rating.append(best_reviews.at[idx, "rating"])
        popularity.append(best_reviews.at[idx, "popularity_rating"])
        landmark.append(best_reviews.at[idx, "landmark"])
        locality.append(best_reviews.at[idx, "locality"])
        url.append(best_reviews.at[idx, "URL"])
        
    hotels = [{"name": n, "city": c + ", " + p, "url": u, "landmark": l,
                    "locality": t, "rating": r} for n, c, p, u, l, t, r 
                  in zip(name, city, country, url, landmark, locality, rating)]
    return hotels

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

def find_similar_hotels(name, weights, index_name = "hotel_name", n = 20, 
                    filtering = False, filter_name = None):
    """ Return json object with most similar hotels """

    hotels_df = get_df()

    # Mapping hotels to integers with get_int_mapping().
    hotel_index, index_hotel, unique_hotels = get_int_mapping(hotels_df, "hotel_name")


    # Select index and reverse index
    if index_name == "hotel_name":
        index = hotel_index
        rindex = index_hotel

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
    closest = sorted_distances[-n:-1]

    # Limit results by filtering
    filter_ = None
    hotel_name = []
    city = []
    country = []
    url = []
    landmark = []
    locality = []
    rating = []

    if filtering:
        for idxs, rows in hotels_df.iterrows():
            if hotels_df.at[idxs, "hotel_name"] == name:
                filter_ = hotels_df.at[idxs, filter_name]
                break
        match_df = hotels_df[hotels_df[filter_name].str.match(filter_)]
        match_df = match_df.reset_index(drop = True)
        match_df["distance"] = None
        for idxs, rows in match_df.iterrows():
            item = match_df.at[idxs, "hotel_name"]
            distance = np.dot(weights[index[item]], weights[index[name]])
            match_df.loc[match_df.index[idxs], "distance"] = distance
        match_df = match_df.sort_values(by = ["distance"], axis = 0, ascending = False)
        to_drop = [name]
        match_df = match_df[~match_df["hotel_name"].isin(to_drop)]
        hotel_name = match_df["hotel_name"].to_list()
        city = match_df["city"].to_list()
        country = match_df["country"].to_list()
        url = match_df["URL"].to_list()
        landmark = match_df["landmark"].to_list()
        locality = match_df["locality"].to_list()
        rating = match_df["rating"].to_list()
        hotels = [{"name": n, "city": c + ", " + p, "url": u, "landmark": l,
                    "locality": t, "rating": r} for n, c, p, u, l, t, r 
                    in zip(hotel_name, city, country, url, landmark, locality, rating)]
        return hotels

    # Create json objects of similar hotels 
    city = []
    country = []
    name = []
    url = []
    landmark = []
    locality = []
    rating = []
    for c in reversed(closest):
        hotel_name = rindex[c]
        match_df = hotels_df[hotels_df["hotel_name"].str.match(hotel_name)]
        for idxs, rows in match_df.iterrows():
            city.append(hotels_df.at[idxs, "city"])
            country.append(hotels_df.at[idxs, "country"])
            name.append(hotels_df.at[idxs, "hotel_name"])
            url.append(hotels_df.at[idxs, "URL"])
            landmark.append(hotels_df.at[idxs, "landmark"])
            locality.append(hotels_df.at[idxs, "locality"])
            rating.append(hotels_df.at[idxs, "rating"])
    hotels = [{"name": n, "city": c + ", " + p, "url": u, "landmark": l,
                    "locality": t, "rating": r} for n, c, p, u, l, t, r 
                    in zip(name, city, country, url, landmark, locality, rating)]
    return hotels
                
def find_similar_cities(name, weights, index_name = "city", n = 20, 
                        filtering = False, filter_name = None):
    """ Return json object with most similar cities """
    
    # Split location string in city and country   
    location_name = name.split(",")
    name = location[0].strip()
    country_name = location[1]

    hotels_df = get_df()

#     # Mapping hotels to integers with get_int_mapping().
    city_index, index_city, unique_cities = get_int_mapping(hotels_df, "city")

   
    # Select index and reverse index
    if index_name == "city":
        index = city_index
        rindex = index_city

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
    closest = sorted_distances[-n:-1]
    
    # Limit results by filtering
    filter_ = None
    hotel_name = []
    city = []
    country = []
    url = []
    landmark = []
    locality = []
    rating = []
    
    if filtering:
        for idxs, rows in hotels_df.iterrows():
            if hotels_df.at[idxs, "city"] == name:
                filter_ = hotels_df.at[idxs, filter_name]
                break
        match_df = hotels_df[hotels_df[filter_name].str.match(filter_)]
        match_df = match_df.reset_index(drop = True)
        match_df["distance"] = None
        for idxs, rows in match_df.iterrows():
            item = match_df.at[idxs, "city"]
            distance = np.dot(weights[index[item]], weights[index[name]])
            match_df.loc[match_df.index[idxs], "distance"] = distance
        match_df = match_df.sort_values(by = ["distance"], axis = 0, ascending = False)
        to_drop = [name]
        match_df = match_df[~match_df["city"].isin(to_drop)]
        hotel_name = match_df["hotel_name"].to_list()
        city = match_df["city"].to_list()
        country = match_df["country"].to_list()
        url = match_df["URL"].to_list()
        landmark = match_df["landmark"].to_list()
        locality = match_df["locality"].to_list()
        rating = match_df["rating"].to_list()
        hotels = [{"name": n, "city": c + ", " + p, "url": u, "landmark": l,
                    "locality": t, "rating": r} for n, c, p, u, l, t, r 
                  in zip(hotel_name, city, country, url, landmark, locality, rating)]
        return hotels
    
    # Create json objects of similar cities 
    city = []
    country = []
    name = []
    url = []
    landmark = []
    locality = []
    rating = []
    for c in reversed(closest):
        city_name = rindex[c]
        match_df = hotels_df[hotels_df["city"].str.match(city_name)]
        for idxs, rows in match_df.iterrows():
            city.append(hotels_df.at[idxs, "city"])
            country.append(hotels_df.at[idxs, "country"])
            name.append(hotels_df.at[idxs, "hotel_name"])
            url.append(hotels_df.at[idxs, "URL"])
            landmark.append(hotels_df.at[idxs, "landmark"])
            locality.append(hotels_df.at[idxs, "locality"])
            rating.append(hotels_df.at[idxs, "rating"])
    hotels = [{"name": n, "city": c + ", " + p, "url": u, "landmark": l,
                    "locality": t, "rating": r} for n, c, p, u, l, t, r 
                  in zip(name, city, country, url, landmark, locality, rating)]
    return hotels





