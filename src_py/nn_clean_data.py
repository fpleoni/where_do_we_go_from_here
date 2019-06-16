
# coding: utf-8

# In[4]:


# Import libraries
from collections import Counter, OrderedDict
# import datawig
from itertools import chain
from keras.layers import Input, Embedding, Add, Reshape, Dense, Multiply, GlobalMaxPool1D, Dot
from keras.models import Model
import numpy as np
import pandas as pd
pd.set_option("display.max_rows", 500)
pd.options.display.max_colwidth = 1000
import random


# In[5]:


# Path to file
hotels_path = "../data/clean_hotels_scraped.csv"

# Dataframe
hotels_df = pd.read_csv(hotels_path, usecols = ["city", "country", "hotel_name", "rating",
                                               "address", "popularity_rating", "locality", "price",
                                               "landmark", "URL"])

# Sanity check
hotels_df.head()


# In[6]:


# Check for null values
hotels_df.isna().sum()


# In[7]:


# Prepare city column

# Convert all cities to lowercase
hotels_df["city"] = hotels_df["city"].apply(lambda x: x.lower())

city_list = hotels_df["city"].tolist()

# Find set of unique cities and convert to a list
unique_cities = list(set(city_list))

# Create indexes for each city
city_index = {city: idx for idx, city in enumerate(unique_cities)}
index_city = {idx: city for city, idx in city_index.items()}


# In[8]:


# Prepare country column

# Convert all countries to lowercase
hotels_df["country"] = hotels_df["country"].apply(lambda x: x.lower())

country_list = hotels_df["country"].tolist()

# Find set of unique countries and convert to a list
unique_countries = list(set(country_list))

# Create indexes for each property
country_index = {country: idx for idx, country in enumerate(unique_countries)}
index_country = {idx: country for country, idx in country_index.items()}


# In[9]:


# Prepare hotel_name column

# Convert all hotels to lowercase
hotels_df["hotel_name"] = hotels_df["hotel_name"].apply(lambda x: x.lower())

# Create hotel names list
hotels_list = hotels_df["hotel_name"].tolist()

# Unique hotels
unique_hotels = list(set(hotels_list))

# Create indexes for each hotel
hotel_index = {hotel: idx for idx, hotel in enumerate(unique_hotels)}
index_hotel = {idx: hotel for hotel, idx in hotel_index.items()}


# In[10]:


# Create ratings list
rating_list = hotels_df["rating"].tolist()

# Find set of unique ratings and convert to a list
unique_ratings = list(set(rating_list))

# Create indexes for each rating
rating_index = {rating: idx for idx, rating in enumerate(unique_ratings)}
index_rating = {idx: rating for rating, idx in rating_index.items()}


# In[11]:


# Create popularity ratings list
popularity_list = hotels_df["popularity_rating"].tolist()

# Find set of unique ratings and convert to a list
unique_popularity = list(set(popularity_list))

# Create indexes for each rating
popularity_index = {popularity: idx for idx, popularity in enumerate(unique_popularity)}
index_popularity = {idx: popularity for popularity, idx in popularity_index.items()}


# In[12]:


# Prepare locality column

# Convert all hotels to lowercase
hotels_df["locality"] = hotels_df["locality"].apply(lambda x: x.lower())

# Create hotel names list
locality_list = hotels_df["locality"].tolist()
unique_localities = list(set(locality_list))

# Create indexes for each hotel
locality_index = {locality: idx for idx, locality in enumerate(unique_localities)}
index_locality = {idx: locality for locality, idx in locality_index.items()}


# In[13]:


# Create price list
price_list = hotels_df["price"].tolist()

# Unique prices
unique_prices = list(set(price_list))

# Create indexes for each price
price_index = {price: idx for idx, price in enumerate(unique_prices)}
index_price = {idx: price for price, idx in price_index.items()}


# In[14]:


# Prepare locality column

# Create hotel names list
landmark_list = hotels_df["landmark"].tolist()

# Find set of unique properties and convert to a list
unique_landmarks = list(chain(*[list(set(landmarks)) for landmarks in landmark_list]))
unique_landmarks = list(set(unique_landmarks))

# Create indexes for each hotel
landmark_index = {landmark: idx for idx, landmark in enumerate(unique_landmarks)}
index_landmark = {idx: landmark for landmark, idx in landmark_index.items()}


# In[15]:


# Build tuples to train embedding neural network
hotel_tuples = []

# Iterate through each row of dataframe
for index, row in hotels_df.iterrows():
    # Iterate through the properties in the item
    hotel_tuples.extend((city_index[hotels_df.at[index, "city"]], country_index[hotels_df.at[index, "country"]],
                         hotel_index[hotels_df.at[index, "hotel_name"]], rating_index[hotels_df.at[index, "rating"]],
                         popularity_index[hotels_df.at[index, "popularity_rating"]], 
                         locality_index[hotels_df.at[index, "locality"]], 
                         price_index[hotels_df.at[index, "price"]], landmark_index[landmark]) for landmark 
                        in hotels_df.at[index, "landmark"] if landmark.lower() in unique_landmarks)


# In[16]:


# Generator for training samples
def generate_batch(tuples, n_positive = 75, negative_ratio = 2.0):
    
    pairs_set = set(tuples)
    
    batch_size = n_positive * (1 + negative_ratio)
    batch = np.zeros((batch_size, 9))
    
    # Label for negative examples
    neg_label = 0
    
    # This creates a generator
    while True:
        # randomly choose positive examples
        for idx, (city_id, country_id, hotel_id, rating_id, popularity_id, locality_id, price_id, landmark_id) in enumerate(random.sample(tuples, n_positive)):
            batch[idx, :] = (city_id, country_id, hotel_id, rating_id, popularity_id, locality_id, price_id, landmark_id, 1)

        # Increment idx by 1
        idx += 1
        
        # Add negative examples until reach batch size
        while idx < batch_size:
            
            # random selection
            random_city = random.randrange(len(unique_cities))
            random_country = random.randrange(len(unique_countries))
            random_hotel = random.randrange(len(unique_hotels))
            random_rating = random.randrange(len(unique_ratings))
            random_popularity = random.randrange(len(unique_popularity))
            random_locality = random.randrange(len(unique_localities))
            random_price = random.randrange(len(unique_prices))
            random_landmark = random.randrange(len(unique_landmarks))
            
            # Check to make sure this is not a positive example
            if (random_city, random_country, random_hotel, random_rating, random_popularity, random_locality,
               random_price, random_landmark) not in pairs_set:
                
                # Add to batch and increment index
                batch[idx, :] = (random_city, random_country, random_hotel, random_rating, random_popularity, random_locality,
                                   random_price, random_landmark, neg_label)
                idx += 1
                
        # Make sure to shuffle order
        np.random.shuffle(batch)
        yield {"city": batch[:, 0], "country": batch[:, 1], "hotel": batch[:, 2], 
               "rating": batch[:, 3], "popularity": batch[:, 4], "locality": batch[:, 5], 
               "price": batch[:, 6], "landmark": batch[:, 7]}, batch[:, 8]


# In[17]:


# Properties embedding model
def hotel_embeddings(embedding_size = 50):
    
    # Inputs are one-dimensional
    city = Input(name = "city", shape = [1])
    country = Input(name = "country", shape = [1])
    hotel = Input(name = "hotel", shape = [1])
    rating = Input(name = "rating", shape = [1])
    popularity = Input(name = "popularity", shape = [1])
    locality = Input(name = "locality", shape = [1])
    price = Input(name = "price", shape = [1])
    landmark = Input(name = "landmark", shape = [1])
    
    # Embedding the city
    city_embedding = Embedding(name = "city_embedding", input_dim = len(city_index), 
                              output_dim = embedding_size)(city)
    
    # Embedding the country
    country_embedding = Embedding(name = "country_embedding", input_dim = len(country_index),
                                  output_dim = embedding_size)(country)
    
    # Embedding the hotel
    hotel_embedding = Embedding(name = "hotel_embedding", input_dim = len(hotel_index),
                                  output_dim = embedding_size)(hotel)
    
    # Embedding the rating
    rating_embedding = Embedding(name = "rating_embedding", input_dim = len(rating_index),
                                  output_dim = embedding_size)(rating)
    
    # Embedding the popularity
    popularity_embedding = Embedding(name = "popularity_embedding", input_dim = len(popularity_index),
                                  output_dim = embedding_size)(popularity)    
    
    # Embedding the locality
    locality_embedding = Embedding(name = "locality_embedding", input_dim = len(locality_index),
                                  output_dim = embedding_size)(locality)
    
    # Embedding the price
    price_embedding = Embedding(name = "price_embedding", input_dim = len(price_index),
                                  output_dim = embedding_size)(price)
    
    # Embedding the landmark
    landmark_embedding = Embedding(name = "landmark_embedding", input_dim = len(landmark_index),
                                  output_dim = embedding_size)(landmark)
    
    
    # Merge the embeddings with multiplication 
    merged_one = Multiply(name = "interaction_one")([city_embedding, country_embedding])
    merged_two = Multiply(name = "interaction_two")([city_embedding, hotel_embedding])
    merged_three = Multiply(name = "interaction_three")([city_embedding, rating_embedding])
    merged_four = Multiply(name = "interaction_four")([city_embedding, locality_embedding])
    merged_five = Multiply(name = "interaction_five")([city_embedding, price_embedding])
    merged_six = Multiply(name = "interaction_six")([city_embedding, landmark_embedding])
    merged_seven = Multiply(name = "interaction_seven")([country_embedding, hotel_embedding])
    merged_eight = Multiply(name = "interaction_eight")([country_embedding, rating_embedding])
    merged_nine = Multiply(name = "interaction_nine")([country_embedding, locality_embedding])
    merged_ten = Multiply(name = "interaction_ten")([country_embedding, price_embedding])
    merged_eleven = Multiply(name = "interaction_eleven")([country_embedding, landmark_embedding])
    merged_twelve = Multiply(name = "interaction_twelve")([hotel_embedding, rating_embedding])  
    merged_thirteen = Multiply(name = "interaction_thirteen")([hotel_embedding, locality_embedding])  
    merged_fourteen = Multiply(name = "interaction_fourteen")([hotel_embedding, price_embedding])  
    merged_fifteen = Multiply(name = "interaction_fifteen")([hotel_embedding, landmark_embedding])  
    merged_sixteen = Multiply(name = "interaction_sixteen")([rating_embedding, locality_embedding])  
    merged_seventeen = Multiply(name = "interaction_seventeen")([rating_embedding, price_embedding])  
    merged_eighteen = Multiply(name = "interaction_eighteen")([rating_embedding, landmark_embedding])  
    merged_nineteen = Multiply(name = "interaction_nineteen")([locality_embedding, price_embedding])  
    merged_twenty = Multiply(name = "interaction_twenty")([locality_embedding, landmark_embedding])
    merged_twentyone = Multiply(name = "interaction_twentyone")([price_embedding, landmark_embedding])
    merged_twentytwo = Multiply(name = "interaction_twentytwo")([popularity_embedding, city_embedding])
    merged_twentythree = Multiply(name = "interaction_twentythree")([popularity_embedding, country_embedding])
    merged_twentyfour = Multiply(name = "interaction_twentyfour")([popularity_embedding, hotel_embedding])
    merged_twentyfive = Multiply(name = "interaction_twentyfive")([popularity_embedding, rating_embedding])
    merged_twentysix = Multiply(name = "interaction_twentysix")([popularity_embedding, locality_embedding])
    merged_twentyseven = Multiply(name = "interaction_twentyseven")([popularity_embedding, price_embedding])
    merged_twentyeight = Multiply(name = "interaction_twentyeight")([popularity_embedding, landmark_embedding])
    
    # GlobalMaxPool
    pooling_one = GlobalMaxPool1D(name = "pooling_one")(merged_one)
    pooling_two = GlobalMaxPool1D(name = "pooling_two")(merged_two)
    pooling_three = GlobalMaxPool1D(name = "pooling_three")(merged_three)
    pooling_four = GlobalMaxPool1D(name = "pooling_four")(merged_four)
    pooling_five = GlobalMaxPool1D(name = "pooling_five")(merged_five)
    pooling_six = GlobalMaxPool1D(name = "pooling_six")(merged_six)    
    pooling_seven = GlobalMaxPool1D(name = "pooling_seven")(merged_seven)
    pooling_eight = GlobalMaxPool1D(name = "pooling_eight")(merged_eight)   
    pooling_nine = GlobalMaxPool1D(name = "pooling_nine")(merged_nine)   
    pooling_ten = GlobalMaxPool1D(name = "pooling_ten")(merged_ten)    
    pooling_eleven = GlobalMaxPool1D(name = "pooling_eleven")(merged_eleven)    
    pooling_twelve = GlobalMaxPool1D(name = "pooling_twelve")(merged_twelve)  
    pooling_thirteen = GlobalMaxPool1D(name = "pooling_thirteen")(merged_thirteen)
    pooling_fourteen = GlobalMaxPool1D(name = "pooling_fourteen")(merged_fourteen)
    pooling_fifteen = GlobalMaxPool1D(name = "pooling_fifteen")(merged_fifteen)
    pooling_sixteen = GlobalMaxPool1D(name = "pooling_sixteen")(merged_sixteen)
    pooling_seventeen = GlobalMaxPool1D(name = "pooling_seventeen")(merged_seventeen)
    pooling_eighteen = GlobalMaxPool1D(name = "pooling_eighteen")(merged_eighteen)
    pooling_nineteen = GlobalMaxPool1D(name = "pooling_nineteen")(merged_nineteen)
    pooling_twenty = GlobalMaxPool1D(name = "pooling_twenty")(merged_twenty)
    pooling_twentyone = GlobalMaxPool1D(name = "pooling_twentyone")(merged_twentyone)
    pooling_twentytwo = GlobalMaxPool1D(name = "pooling_twentytwo")(merged_twentytwo)
    pooling_twentythree = GlobalMaxPool1D(name = "pooling_twentythree")(merged_twentythree)
    pooling_twentyfour = GlobalMaxPool1D(name = "pooling_twentyfour")(merged_twentyfour)
    pooling_twentyfive = GlobalMaxPool1D(name = "pooling_twentyfive")(merged_twentyfive)
    pooling_twentysix = GlobalMaxPool1D(name = "pooling_twentysix")(merged_twentysix)
    pooling_twentyseven = GlobalMaxPool1D(name = "pooling_twentyseven")(merged_twentyseven)
    pooling_twentyeight = GlobalMaxPool1D(name = "pooling_twentyeight")(merged_twentyeight)
    
    # Dot Product
    dot_1 = Dot(normalize = True, axes = -1)([pooling_one, pooling_two])
    dot_2 = Dot(normalize = True, axes = -1)([pooling_one, pooling_three])
    dot_3 = Dot(normalize = True, axes = -1)([pooling_one, pooling_four])
    dot_4 = Dot(normalize = True, axes = -1)([pooling_one, pooling_five])
    dot_5 = Dot(normalize = True, axes = -1)([pooling_one, pooling_six])
    dot_6 = Dot(normalize = True, axes = -1)([pooling_one, pooling_seven])
    dot_7 = Dot(normalize = True, axes = -1)([pooling_one, pooling_eight])
    dot_8 = Dot(normalize = True, axes = -1)([pooling_one, pooling_nine])
    dot_9 = Dot(normalize = True, axes = -1)([pooling_one, pooling_ten])
    dot_10 = Dot(normalize = True, axes = -1)([pooling_one, pooling_eleven])
    dot_11 = Dot(normalize = True, axes = -1)([pooling_one, pooling_twelve])
    dot_12 = Dot(normalize = True, axes = -1)([pooling_one, pooling_thirteen])
    dot_13 = Dot(normalize = True, axes = -1)([pooling_one, pooling_fourteen])
    dot_14 = Dot(normalize = True, axes = -1)([pooling_one, pooling_fifteen])
    dot_15 = Dot(normalize = True, axes = -1)([pooling_one, pooling_sixteen])
    dot_16 = Dot(normalize = True, axes = -1)([pooling_one, pooling_seventeen])
    dot_17 = Dot(normalize = True, axes = -1)([pooling_one, pooling_eighteen])
    dot_18 = Dot(normalize = True, axes = -1)([pooling_one, pooling_nineteen])
    dot_19 = Dot(normalize = True, axes = -1)([pooling_one, pooling_twenty])
    dot_20 = Dot(normalize = True, axes = -1)([pooling_one, pooling_twentyone])
    dot_21 = Dot(normalize = True, axes = -1)([pooling_one, pooling_twentytwo])
    dot_22 = Dot(normalize = True, axes = -1)([pooling_one, pooling_twentythree])
    dot_23 = Dot(normalize = True, axes = -1)([pooling_one, pooling_twentyfour])
    dot_24 = Dot(normalize = True, axes = -1)([pooling_one, pooling_twentyfive])    
    dot_25 = Dot(normalize = True, axes = -1)([pooling_one, pooling_twentysix])    
    dot_26 = Dot(normalize = True, axes = -1)([pooling_one, pooling_twentyseven])    
    dot_27 = Dot(normalize = True, axes = -1)([pooling_one, pooling_twentyeight])    
    dot_28 = Dot(normalize = True, axes = -1)([pooling_two, pooling_three])    
    dot_29 = Dot(normalize = True, axes = -1)([pooling_two, pooling_four])   
    dot_30 = Dot(normalize = True, axes = -1)([pooling_two, pooling_five])   
    dot_31 = Dot(normalize = True, axes = -1)([pooling_two, pooling_six])   
    dot_32 = Dot(normalize = True, axes = -1)([pooling_two, pooling_seven])   
    dot_33 = Dot(normalize = True, axes = -1)([pooling_two, pooling_eight])   
    dot_34 = Dot(normalize = True, axes = -1)([pooling_two, pooling_nine])   
    dot_35 = Dot(normalize = True, axes = -1)([pooling_two, pooling_ten])   
    dot_36 = Dot(normalize = True, axes = -1)([pooling_two, pooling_eleven])   
    dot_37 = Dot(normalize = True, axes = -1)([pooling_two, pooling_twelve])   
    dot_38 = Dot(normalize = True, axes = -1)([pooling_two, pooling_thirteen])   
    dot_39 = Dot(normalize = True, axes = -1)([pooling_two, pooling_fourteen])   
    dot_40 = Dot(normalize = True, axes = -1)([pooling_two, pooling_fifteen])   
    dot_41 = Dot(normalize = True, axes = -1)([pooling_two, pooling_sixteen])   
    dot_42 = Dot(normalize = True, axes = -1)([pooling_two, pooling_seventeen])   
    dot_43 = Dot(normalize = True, axes = -1)([pooling_two, pooling_eighteen]) 
    dot_44 = Dot(normalize = True, axes = -1)([pooling_two, pooling_nineteen]) 
    dot_45 = Dot(normalize = True, axes = -1)([pooling_two, pooling_twenty]) 
    dot_46 = Dot(normalize = True, axes = -1)([pooling_two, pooling_twentyone]) 
    dot_47 = Dot(normalize = True, axes = -1)([pooling_two, pooling_twentytwo]) 
    dot_48 = Dot(normalize = True, axes = -1)([pooling_two, pooling_twentythree]) 
    dot_49 = Dot(normalize = True, axes = -1)([pooling_two, pooling_twentyfour]) 
    dot_50 = Dot(normalize = True, axes = -1)([pooling_two, pooling_twentyfive]) 
    dot_51 = Dot(normalize = True, axes = -1)([pooling_two, pooling_twentysix]) 
    dot_52 = Dot(normalize = True, axes = -1)([pooling_two, pooling_twentyseven]) 
    dot_53 = Dot(normalize = True, axes = -1)([pooling_two, pooling_twentyeight]) 
    dot_54 = Dot(normalize = True, axes = -1)([pooling_three, pooling_four])    
    dot_55 = Dot(normalize = True, axes = -1)([pooling_three, pooling_five]) 
    dot_56 = Dot(normalize = True, axes = -1)([pooling_three, pooling_six]) 
    dot_57 = Dot(normalize = True, axes = -1)([pooling_three, pooling_seven]) 
    dot_58 = Dot(normalize = True, axes = -1)([pooling_three, pooling_eight]) 
    dot_59 = Dot(normalize = True, axes = -1)([pooling_three, pooling_nine]) 
    dot_60 = Dot(normalize = True, axes = -1)([pooling_three, pooling_ten]) 
    dot_61 = Dot(normalize = True, axes = -1)([pooling_three, pooling_eleven]) 
    dot_62 = Dot(normalize = True, axes = -1)([pooling_three, pooling_twelve]) 
    dot_63 = Dot(normalize = True, axes = -1)([pooling_three, pooling_thirteen]) 
    dot_64 = Dot(normalize = True, axes = -1)([pooling_three, pooling_fourteen]) 
    dot_65 = Dot(normalize = True, axes = -1)([pooling_three, pooling_fifteen]) 
    dot_66 = Dot(normalize = True, axes = -1)([pooling_three, pooling_sixteen]) 
    dot_67 = Dot(normalize = True, axes = -1)([pooling_three, pooling_seventeen]) 
    dot_68 = Dot(normalize = True, axes = -1)([pooling_three, pooling_eighteen]) 
    dot_69 = Dot(normalize = True, axes = -1)([pooling_three, pooling_nineteen]) 
    dot_70 = Dot(normalize = True, axes = -1)([pooling_three, pooling_twenty])
    dot_71 = Dot(normalize = True, axes = -1)([pooling_three, pooling_twentyone])
    dot_72 = Dot(normalize = True, axes = -1)([pooling_three, pooling_twentytwo])
    dot_73 = Dot(normalize = True, axes = -1)([pooling_three, pooling_twentythree])
    dot_74 = Dot(normalize = True, axes = -1)([pooling_three, pooling_twentyfour])
    dot_75 = Dot(normalize = True, axes = -1)([pooling_three, pooling_twentyfive])
    dot_76 = Dot(normalize = True, axes = -1)([pooling_three, pooling_twentysix])
    dot_77 = Dot(normalize = True, axes = -1)([pooling_three, pooling_twentyseven])
    dot_78 = Dot(normalize = True, axes = -1)([pooling_three, pooling_twentyeight])
    dot_79 = Dot(normalize = True, axes = -1)([pooling_four, pooling_five])
    dot_80 = Dot(normalize = True, axes = -1)([pooling_four, pooling_six])
    dot_81 = Dot(normalize = True, axes = -1)([pooling_four, pooling_seven])
    dot_82 = Dot(normalize = True, axes = -1)([pooling_four, pooling_eight])
    dot_83 = Dot(normalize = True, axes = -1)([pooling_four, pooling_nine])
    dot_84 = Dot(normalize = True, axes = -1)([pooling_four, pooling_ten])
    dot_85 = Dot(normalize = True, axes = -1)([pooling_four, pooling_eleven])
    dot_86 = Dot(normalize = True, axes = -1)([pooling_four, pooling_twelve])
    dot_87 = Dot(normalize = True, axes = -1)([pooling_four, pooling_thirteen])
    dot_88 = Dot(normalize = True, axes = -1)([pooling_four, pooling_fourteen])
    dot_89 = Dot(normalize = True, axes = -1)([pooling_four, pooling_fifteen])
    dot_90 = Dot(normalize = True, axes = -1)([pooling_four, pooling_sixteen])
    dot_91 = Dot(normalize = True, axes = -1)([pooling_four, pooling_seventeen])
    dot_92 = Dot(normalize = True, axes = -1)([pooling_four, pooling_eighteen])
    dot_93 = Dot(normalize = True, axes = -1)([pooling_four, pooling_nineteen])
    dot_94 = Dot(normalize = True, axes = -1)([pooling_four, pooling_twenty])
    dot_95 = Dot(normalize = True, axes = -1)([pooling_four, pooling_twentyone])
    dot_96 = Dot(normalize = True, axes = -1)([pooling_four, pooling_twentytwo])
    dot_97 = Dot(normalize = True, axes = -1)([pooling_four, pooling_twentythree])
    dot_98 = Dot(normalize = True, axes = -1)([pooling_four, pooling_twentyfour])
    dot_99 = Dot(normalize = True, axes = -1)([pooling_four, pooling_twentyfive])
    dot_100 = Dot(normalize = True, axes = -1)([pooling_four, pooling_twentysix])
    dot_101 = Dot(normalize = True, axes = -1)([pooling_four, pooling_twentyseven])
    dot_102 = Dot(normalize = True, axes = -1)([pooling_four, pooling_twentyeight])
    dot_103 = Dot(normalize = True, axes = -1)([pooling_five, pooling_six])
    dot_104 = Dot(normalize = True, axes = -1)([pooling_five, pooling_seven])
    dot_105 = Dot(normalize = True, axes = -1)([pooling_five, pooling_eight])
    dot_106 = Dot(normalize = True, axes = -1)([pooling_five, pooling_nine])
    dot_107 = Dot(normalize = True, axes = -1)([pooling_five, pooling_ten])
    dot_108 = Dot(normalize = True, axes = -1)([pooling_five, pooling_eleven])
    dot_109 = Dot(normalize = True, axes = -1)([pooling_five, pooling_twelve])
    dot_110 = Dot(normalize = True, axes = -1)([pooling_five, pooling_thirteen])
    dot_111 = Dot(normalize = True, axes = -1)([pooling_five, pooling_fourteen])
    dot_112 = Dot(normalize = True, axes = -1)([pooling_five, pooling_fifteen])
    dot_113 = Dot(normalize = True, axes = -1)([pooling_five, pooling_sixteen])
    dot_114 = Dot(normalize = True, axes = -1)([pooling_five, pooling_seventeen])
    dot_116 = Dot(normalize = True, axes = -1)([pooling_five, pooling_eighteen])
    dot_117 = Dot(normalize = True, axes = -1)([pooling_five, pooling_nineteen])
    dot_118 = Dot(normalize = True, axes = -1)([pooling_five, pooling_twenty])
    dot_119 = Dot(normalize = True, axes = -1)([pooling_five, pooling_twentyone])
    dot_371 = Dot(normalize = True, axes = -1)([pooling_five, pooling_twentytwo])
    dot_372 = Dot(normalize = True, axes = -1)([pooling_five, pooling_twentythree])
    dot_373 = Dot(normalize = True, axes = -1)([pooling_five, pooling_twentyfour])
    dot_374 = Dot(normalize = True, axes = -1)([pooling_five, pooling_twentyfive])
    dot_375 = Dot(normalize = True, axes = -1)([pooling_five, pooling_twentysix])
    dot_376 = Dot(normalize = True, axes = -1)([pooling_five, pooling_twentyseven])
    dot_377 = Dot(normalize = True, axes = -1)([pooling_five, pooling_twentyeight])
    dot_378 = Dot(normalize = True, axes = -1)([pooling_six, pooling_seven])
    dot_379 = Dot(normalize = True, axes = -1)([pooling_six, pooling_eight])
    dot_380 = Dot(normalize = True, axes = -1)([pooling_six, pooling_nine])
    dot_381 = Dot(normalize = True, axes = -1)([pooling_six, pooling_ten])
    dot_382 = Dot(normalize = True, axes = -1)([pooling_six, pooling_eleven])
    dot_383 = Dot(normalize = True, axes = -1)([pooling_six, pooling_twelve])
    dot_384 = Dot(normalize = True, axes = -1)([pooling_six, pooling_thirteen])
    dot_385 = Dot(normalize = True, axes = -1)([pooling_six, pooling_fourteen])
    dot_386 = Dot(normalize = True, axes = -1)([pooling_six, pooling_fifteen])
    dot_387 = Dot(normalize = True, axes = -1)([pooling_six, pooling_sixteen])
    dot_388 = Dot(normalize = True, axes = -1)([pooling_six, pooling_seventeen])
    dot_389 = Dot(normalize = True, axes = -1)([pooling_six, pooling_eighteen])
    dot_391 = Dot(normalize = True, axes = -1)([pooling_six, pooling_nineteen])
    dot_392 = Dot(normalize = True, axes = -1)([pooling_six, pooling_twenty])
    dot_393 = Dot(normalize = True, axes = -1)([pooling_six, pooling_twentyone])
    dot_394 = Dot(normalize = True, axes = -1)([pooling_six, pooling_twentytwo])
    dot_395 = Dot(normalize = True, axes = -1)([pooling_six, pooling_twentythree])
    dot_120 = Dot(normalize = True, axes = -1)([pooling_six, pooling_twentyfour])
    dot_121 = Dot(normalize = True, axes = -1)([pooling_six, pooling_twentyfive])
    dot_122 = Dot(normalize = True, axes = -1)([pooling_six, pooling_twentysix])
    dot_123 = Dot(normalize = True, axes = -1)([pooling_six, pooling_twentyseven])
    dot_124 = Dot(normalize = True, axes = -1)([pooling_six, pooling_twentyeight])
    dot_125 = Dot(normalize = True, axes = -1)([pooling_seven, pooling_eight])
    dot_126 = Dot(normalize = True, axes = -1)([pooling_seven, pooling_nine])
    dot_127 = Dot(normalize = True, axes = -1)([pooling_seven, pooling_ten])
    dot_128 = Dot(normalize = True, axes = -1)([pooling_seven, pooling_eleven])
    dot_129 = Dot(normalize = True, axes = -1)([pooling_seven, pooling_twelve])
    dot_130 = Dot(normalize = True, axes = -1)([pooling_seven, pooling_thirteen])
    dot_131 = Dot(normalize = True, axes = -1)([pooling_seven, pooling_fourteen])
    dot_132 = Dot(normalize = True, axes = -1)([pooling_seven, pooling_fifteen])
    dot_133 = Dot(normalize = True, axes = -1)([pooling_seven, pooling_sixteen])
    dot_134 = Dot(normalize = True, axes = -1)([pooling_seven, pooling_seventeen])
    dot_135 = Dot(normalize = True, axes = -1)([pooling_seven, pooling_eighteen])
    dot_136 = Dot(normalize = True, axes = -1)([pooling_seven, pooling_nineteen])
    dot_137 = Dot(normalize = True, axes = -1)([pooling_seven, pooling_twenty])
    dot_138 = Dot(normalize = True, axes = -1)([pooling_seven, pooling_twentyone])
    dot_139 = Dot(normalize = True, axes = -1)([pooling_seven, pooling_twentytwo])
    dot_140 = Dot(normalize = True, axes = -1)([pooling_seven, pooling_twentythree])
    dot_141 = Dot(normalize = True, axes = -1)([pooling_seven, pooling_twentyfour])
    dot_142 = Dot(normalize = True, axes = -1)([pooling_seven, pooling_twentyfive])
    dot_143 = Dot(normalize = True, axes = -1)([pooling_seven, pooling_twentysix])
    dot_144 = Dot(normalize = True, axes = -1)([pooling_seven, pooling_twentyseven])
    dot_145 = Dot(normalize = True, axes = -1)([pooling_seven, pooling_twentyeight])
    dot_146 = Dot(normalize = True, axes = -1)([pooling_eight, pooling_nine])
    dot_147 = Dot(normalize = True, axes = -1)([pooling_eight, pooling_ten])
    dot_148 = Dot(normalize = True, axes = -1)([pooling_eight, pooling_eleven])
    dot_149 = Dot(normalize = True, axes = -1)([pooling_eight, pooling_twelve])
    dot_150 = Dot(normalize = True, axes = -1)([pooling_eight, pooling_thirteen])
    dot_151 = Dot(normalize = True, axes = -1)([pooling_eight, pooling_fourteen])
    dot_152 = Dot(normalize = True, axes = -1)([pooling_eight, pooling_fifteen])
    dot_153 = Dot(normalize = True, axes = -1)([pooling_eight, pooling_sixteen])
    dot_154 = Dot(normalize = True, axes = -1)([pooling_eight, pooling_seventeen])
    dot_155 = Dot(normalize = True, axes = -1)([pooling_eight, pooling_eighteen])
    dot_156 = Dot(normalize = True, axes = -1)([pooling_eight, pooling_nineteen])
    dot_157 = Dot(normalize = True, axes = -1)([pooling_eight, pooling_twenty])
    dot_158 = Dot(normalize = True, axes = -1)([pooling_eight, pooling_twentyone])
    dot_159 = Dot(normalize = True, axes = -1)([pooling_eight, pooling_twentytwo])
    dot_160 = Dot(normalize = True, axes = -1)([pooling_eight, pooling_twentythree])
    dot_161 = Dot(normalize = True, axes = -1)([pooling_eight, pooling_twentyfour])
    dot_162 = Dot(normalize = True, axes = -1)([pooling_eight, pooling_twentyfive])
    dot_163 = Dot(normalize = True, axes = -1)([pooling_eight, pooling_twentysix])
    dot_164 = Dot(normalize = True, axes = -1)([pooling_eight, pooling_twentyseven])
    dot_165 = Dot(normalize = True, axes = -1)([pooling_eight, pooling_twentyeight])
    dot_167 = Dot(normalize = True, axes = -1)([pooling_nine, pooling_ten])
    dot_168 = Dot(normalize = True, axes = -1)([pooling_nine, pooling_eleven])
    dot_169 = Dot(normalize = True, axes = -1)([pooling_nine, pooling_twelve])
    dot_170 = Dot(normalize = True, axes = -1)([pooling_nine, pooling_thirteen])
    dot_171 = Dot(normalize = True, axes = -1)([pooling_nine, pooling_fourteen])
    dot_172 = Dot(normalize = True, axes = -1)([pooling_nine, pooling_fifteen])
    dot_173 = Dot(normalize = True, axes = -1)([pooling_nine, pooling_sixteen])
    dot_174 = Dot(normalize = True, axes = -1)([pooling_nine, pooling_seventeen])
    dot_175 = Dot(normalize = True, axes = -1)([pooling_nine, pooling_eighteen])
    dot_176 = Dot(normalize = True, axes = -1)([pooling_nine, pooling_nineteen])
    dot_177 = Dot(normalize = True, axes = -1)([pooling_nine, pooling_twenty])
    dot_178 = Dot(normalize = True, axes = -1)([pooling_nine, pooling_twentyone])
    dot_179 = Dot(normalize = True, axes = -1)([pooling_nine, pooling_twentytwo])
    dot_180 = Dot(normalize = True, axes = -1)([pooling_nine, pooling_twentythree])
    dot_181 = Dot(normalize = True, axes = -1)([pooling_nine, pooling_twentyfour])
    dot_182 = Dot(normalize = True, axes = -1)([pooling_nine, pooling_twentyfive])
    dot_183 = Dot(normalize = True, axes = -1)([pooling_nine, pooling_twentysix])
    dot_184 = Dot(normalize = True, axes = -1)([pooling_nine, pooling_twentyseven])
    dot_185 = Dot(normalize = True, axes = -1)([pooling_nine, pooling_twentyeight])
    dot_187 = Dot(normalize = True, axes = -1)([pooling_ten, pooling_eleven])
    dot_188 = Dot(normalize = True, axes = -1)([pooling_ten, pooling_twelve])
    dot_189 = Dot(normalize = True, axes = -1)([pooling_ten, pooling_thirteen])
    dot_190 = Dot(normalize = True, axes = -1)([pooling_ten, pooling_fourteen])
    dot_191 = Dot(normalize = True, axes = -1)([pooling_ten, pooling_fifteen])
    dot_192 = Dot(normalize = True, axes = -1)([pooling_ten, pooling_sixteen])
    dot_193 = Dot(normalize = True, axes = -1)([pooling_ten, pooling_seventeen])
    dot_194 = Dot(normalize = True, axes = -1)([pooling_ten, pooling_eighteen])
    dot_195 = Dot(normalize = True, axes = -1)([pooling_ten, pooling_nineteen])
    dot_196 = Dot(normalize = True, axes = -1)([pooling_ten, pooling_twenty])
    dot_197 = Dot(normalize = True, axes = -1)([pooling_ten, pooling_twentyone])
    dot_198 = Dot(normalize = True, axes = -1)([pooling_ten, pooling_twentytwo])
    dot_199 = Dot(normalize = True, axes = -1)([pooling_ten, pooling_twentythree])
    dot_200 = Dot(normalize = True, axes = -1)([pooling_ten, pooling_twentyfour])
    dot_201 = Dot(normalize = True, axes = -1)([pooling_ten, pooling_twentyfive])
    dot_202 = Dot(normalize = True, axes = -1)([pooling_ten, pooling_twentysix])
    dot_203 = Dot(normalize = True, axes = -1)([pooling_ten, pooling_twentyseven])
    dot_204 = Dot(normalize = True, axes = -1)([pooling_ten, pooling_twentyeight])
    dot_206 = Dot(normalize = True, axes = -1)([pooling_eleven, pooling_twelve])
    dot_207 = Dot(normalize = True, axes = -1)([pooling_eleven, pooling_thirteen])
    dot_208 = Dot(normalize = True, axes = -1)([pooling_eleven, pooling_fourteen])
    dot_209 = Dot(normalize = True, axes = -1)([pooling_eleven, pooling_fifteen])
    dot_210 = Dot(normalize = True, axes = -1)([pooling_eleven, pooling_sixteen])
    dot_211 = Dot(normalize = True, axes = -1)([pooling_eleven, pooling_seventeen])
    dot_212 = Dot(normalize = True, axes = -1)([pooling_eleven, pooling_eighteen])
    dot_213 = Dot(normalize = True, axes = -1)([pooling_eleven, pooling_nineteen])
    dot_214 = Dot(normalize = True, axes = -1)([pooling_eleven, pooling_twenty])
    dot_215 = Dot(normalize = True, axes = -1)([pooling_eleven, pooling_twentyone])
    dot_216 = Dot(normalize = True, axes = -1)([pooling_eleven, pooling_twentytwo])
    dot_217 = Dot(normalize = True, axes = -1)([pooling_eleven, pooling_twentythree])
    dot_218 = Dot(normalize = True, axes = -1)([pooling_eleven, pooling_twentyfour])
    dot_219 = Dot(normalize = True, axes = -1)([pooling_eleven, pooling_twentyfive])
    dot_220 = Dot(normalize = True, axes = -1)([pooling_eleven, pooling_twentysix])
    dot_221 = Dot(normalize = True, axes = -1)([pooling_eleven, pooling_twentyseven])
    dot_222 = Dot(normalize = True, axes = -1)([pooling_eleven, pooling_twentyeight])   
    dot_224 = Dot(normalize = True, axes = -1)([pooling_twelve, pooling_thirteen])
    dot_225 = Dot(normalize = True, axes = -1)([pooling_twelve, pooling_fourteen])
    dot_226 = Dot(normalize = True, axes = -1)([pooling_twelve, pooling_fifteen])
    dot_227 = Dot(normalize = True, axes = -1)([pooling_twelve, pooling_sixteen])
    dot_228 = Dot(normalize = True, axes = -1)([pooling_twelve, pooling_seventeen])
    dot_229 = Dot(normalize = True, axes = -1)([pooling_twelve, pooling_eighteen])
    dot_230 = Dot(normalize = True, axes = -1)([pooling_twelve, pooling_nineteen])
    dot_231 = Dot(normalize = True, axes = -1)([pooling_twelve, pooling_twenty])
    dot_232 = Dot(normalize = True, axes = -1)([pooling_twelve, pooling_twentyone])
    dot_233 = Dot(normalize = True, axes = -1)([pooling_twelve, pooling_twentytwo])
    dot_234 = Dot(normalize = True, axes = -1)([pooling_twelve, pooling_twentythree])
    dot_235 = Dot(normalize = True, axes = -1)([pooling_twelve, pooling_twentyfour])
    dot_236 = Dot(normalize = True, axes = -1)([pooling_twelve, pooling_twentyfive])
    dot_237 = Dot(normalize = True, axes = -1)([pooling_twelve, pooling_twentysix])
    dot_238 = Dot(normalize = True, axes = -1)([pooling_twelve, pooling_twentyseven])
    dot_239 = Dot(normalize = True, axes = -1)([pooling_twelve, pooling_twentyeight])    
    dot_241 = Dot(normalize = True, axes = -1)([pooling_thirteen, pooling_fourteen])
    dot_242 = Dot(normalize = True, axes = -1)([pooling_thirteen, pooling_fifteen])
    dot_243 = Dot(normalize = True, axes = -1)([pooling_thirteen, pooling_sixteen])
    dot_244 = Dot(normalize = True, axes = -1)([pooling_thirteen, pooling_seventeen])
    dot_245 = Dot(normalize = True, axes = -1)([pooling_thirteen, pooling_eighteen])
    dot_246 = Dot(normalize = True, axes = -1)([pooling_thirteen, pooling_nineteen])
    dot_247 = Dot(normalize = True, axes = -1)([pooling_thirteen, pooling_twenty])
    dot_248 = Dot(normalize = True, axes = -1)([pooling_thirteen, pooling_twentyone])
    dot_249 = Dot(normalize = True, axes = -1)([pooling_thirteen, pooling_twentytwo])
    dot_250 = Dot(normalize = True, axes = -1)([pooling_thirteen, pooling_twentythree])
    dot_251 = Dot(normalize = True, axes = -1)([pooling_thirteen, pooling_twentyfour])
    dot_252 = Dot(normalize = True, axes = -1)([pooling_thirteen, pooling_twentyfive])
    dot_253 = Dot(normalize = True, axes = -1)([pooling_thirteen, pooling_twentysix])
    dot_254 = Dot(normalize = True, axes = -1)([pooling_thirteen, pooling_twentyseven])
    dot_255 = Dot(normalize = True, axes = -1)([pooling_thirteen, pooling_twentyeight]) 
    dot_257 = Dot(normalize = True, axes = -1)([pooling_fourteen, pooling_fifteen])
    dot_258 = Dot(normalize = True, axes = -1)([pooling_fourteen, pooling_sixteen])
    dot_259 = Dot(normalize = True, axes = -1)([pooling_fourteen, pooling_seventeen])
    dot_260 = Dot(normalize = True, axes = -1)([pooling_fourteen, pooling_eighteen])
    dot_261 = Dot(normalize = True, axes = -1)([pooling_fourteen, pooling_nineteen])
    dot_262 = Dot(normalize = True, axes = -1)([pooling_fourteen, pooling_twenty])
    dot_263 = Dot(normalize = True, axes = -1)([pooling_fourteen, pooling_twentyone])
    dot_264 = Dot(normalize = True, axes = -1)([pooling_fourteen, pooling_twentytwo])
    dot_265 = Dot(normalize = True, axes = -1)([pooling_fourteen, pooling_twentythree])
    dot_266 = Dot(normalize = True, axes = -1)([pooling_fourteen, pooling_twentyfour])
    dot_267 = Dot(normalize = True, axes = -1)([pooling_fourteen, pooling_twentyfive])
    dot_268 = Dot(normalize = True, axes = -1)([pooling_fourteen, pooling_twentysix])
    dot_269 = Dot(normalize = True, axes = -1)([pooling_fourteen, pooling_twentyseven])
    dot_270 = Dot(normalize = True, axes = -1)([pooling_fourteen, pooling_twentyeight])
    dot_272 = Dot(normalize = True, axes = -1)([pooling_fifteen, pooling_sixteen])
    dot_273 = Dot(normalize = True, axes = -1)([pooling_fifteen, pooling_seventeen])
    dot_274 = Dot(normalize = True, axes = -1)([pooling_fifteen, pooling_eighteen])
    dot_275 = Dot(normalize = True, axes = -1)([pooling_fifteen, pooling_nineteen])
    dot_276 = Dot(normalize = True, axes = -1)([pooling_fifteen, pooling_twenty])
    dot_277 = Dot(normalize = True, axes = -1)([pooling_fifteen, pooling_twentyone])
    dot_278 = Dot(normalize = True, axes = -1)([pooling_fifteen, pooling_twentytwo])
    dot_279 = Dot(normalize = True, axes = -1)([pooling_fifteen, pooling_twentythree])
    dot_280 = Dot(normalize = True, axes = -1)([pooling_fifteen, pooling_twentyfour])
    dot_281 = Dot(normalize = True, axes = -1)([pooling_fifteen, pooling_twentyfive])
    dot_282 = Dot(normalize = True, axes = -1)([pooling_fifteen, pooling_twentysix])
    dot_283 = Dot(normalize = True, axes = -1)([pooling_fifteen, pooling_twentyseven])
    dot_284 = Dot(normalize = True, axes = -1)([pooling_fifteen, pooling_twentyeight])
    dot_286 = Dot(normalize = True, axes = -1)([pooling_sixteen, pooling_seventeen])
    dot_287 = Dot(normalize = True, axes = -1)([pooling_sixteen, pooling_eighteen])
    dot_288 = Dot(normalize = True, axes = -1)([pooling_sixteen, pooling_nineteen])
    dot_289 = Dot(normalize = True, axes = -1)([pooling_sixteen, pooling_twenty])
    dot_290 = Dot(normalize = True, axes = -1)([pooling_sixteen, pooling_twentyone])
    dot_291 = Dot(normalize = True, axes = -1)([pooling_sixteen, pooling_twentytwo])
    dot_292 = Dot(normalize = True, axes = -1)([pooling_sixteen, pooling_twentythree])
    dot_293 = Dot(normalize = True, axes = -1)([pooling_sixteen, pooling_twentyfour])
    dot_294 = Dot(normalize = True, axes = -1)([pooling_sixteen, pooling_twentyfive])
    dot_295 = Dot(normalize = True, axes = -1)([pooling_sixteen, pooling_twentysix])
    dot_296 = Dot(normalize = True, axes = -1)([pooling_sixteen, pooling_twentyseven])
    dot_297 = Dot(normalize = True, axes = -1)([pooling_sixteen, pooling_twentyeight])
    dot_299 = Dot(normalize = True, axes = -1)([pooling_seventeen, pooling_eighteen])
    dot_300 = Dot(normalize = True, axes = -1)([pooling_seventeen, pooling_nineteen])
    dot_301 = Dot(normalize = True, axes = -1)([pooling_seventeen, pooling_twenty])
    dot_302 = Dot(normalize = True, axes = -1)([pooling_seventeen, pooling_twentyone])
    dot_303 = Dot(normalize = True, axes = -1)([pooling_seventeen, pooling_twentytwo])
    dot_304 = Dot(normalize = True, axes = -1)([pooling_seventeen, pooling_twentythree])
    dot_305 = Dot(normalize = True, axes = -1)([pooling_seventeen, pooling_twentyfour])
    dot_306 = Dot(normalize = True, axes = -1)([pooling_seventeen, pooling_twentyfive])
    dot_307 = Dot(normalize = True, axes = -1)([pooling_seventeen, pooling_twentysix])
    dot_308 = Dot(normalize = True, axes = -1)([pooling_seventeen, pooling_twentyseven])
    dot_309 = Dot(normalize = True, axes = -1)([pooling_seventeen, pooling_twentyeight]) 
    dot_311 = Dot(normalize = True, axes = -1)([pooling_eighteen, pooling_nineteen])
    dot_312 = Dot(normalize = True, axes = -1)([pooling_eighteen, pooling_twenty])
    dot_313 = Dot(normalize = True, axes = -1)([pooling_eighteen, pooling_twentyone])
    dot_314 = Dot(normalize = True, axes = -1)([pooling_eighteen, pooling_twentytwo])
    dot_315 = Dot(normalize = True, axes = -1)([pooling_eighteen, pooling_twentythree])
    dot_316 = Dot(normalize = True, axes = -1)([pooling_eighteen, pooling_twentyfour])
    dot_317 = Dot(normalize = True, axes = -1)([pooling_eighteen, pooling_twentyfive])
    dot_318 = Dot(normalize = True, axes = -1)([pooling_eighteen, pooling_twentysix])
    dot_319 = Dot(normalize = True, axes = -1)([pooling_eighteen, pooling_twentyseven])
    dot_320 = Dot(normalize = True, axes = -1)([pooling_eighteen, pooling_twentyeight])
    dot_322 = Dot(normalize = True, axes = -1)([pooling_nineteen, pooling_twenty])
    dot_323 = Dot(normalize = True, axes = -1)([pooling_nineteen, pooling_twentyone])
    dot_324 = Dot(normalize = True, axes = -1)([pooling_nineteen, pooling_twentytwo])
    dot_325 = Dot(normalize = True, axes = -1)([pooling_nineteen, pooling_twentythree])
    dot_326 = Dot(normalize = True, axes = -1)([pooling_nineteen, pooling_twentyfour])
    dot_327 = Dot(normalize = True, axes = -1)([pooling_nineteen, pooling_twentyfive])
    dot_328 = Dot(normalize = True, axes = -1)([pooling_nineteen, pooling_twentysix])
    dot_329 = Dot(normalize = True, axes = -1)([pooling_nineteen, pooling_twentyseven])
    dot_330 = Dot(normalize = True, axes = -1)([pooling_nineteen, pooling_twentyeight])
    dot_332 = Dot(normalize = True, axes = -1)([pooling_twenty, pooling_twentyone])
    dot_333 = Dot(normalize = True, axes = -1)([pooling_twenty, pooling_twentytwo])
    dot_334 = Dot(normalize = True, axes = -1)([pooling_twenty, pooling_twentythree])
    dot_335 = Dot(normalize = True, axes = -1)([pooling_twenty, pooling_twentyfour])
    dot_336 = Dot(normalize = True, axes = -1)([pooling_twenty, pooling_twentyfive])
    dot_337 = Dot(normalize = True, axes = -1)([pooling_twenty, pooling_twentysix])
    dot_338 = Dot(normalize = True, axes = -1)([pooling_twenty, pooling_twentyseven])
    dot_339 = Dot(normalize = True, axes = -1)([pooling_twenty, pooling_twentyeight])
    dot_341 = Dot(normalize = True, axes = -1)([pooling_twentyone, pooling_twentytwo])
    dot_342 = Dot(normalize = True, axes = -1)([pooling_twentyone, pooling_twentythree])
    dot_343 = Dot(normalize = True, axes = -1)([pooling_twentyone, pooling_twentyfour])
    dot_344 = Dot(normalize = True, axes = -1)([pooling_twentyone, pooling_twentyfive])
    dot_345 = Dot(normalize = True, axes = -1)([pooling_twentyone, pooling_twentysix])
    dot_346 = Dot(normalize = True, axes = -1)([pooling_twentyone, pooling_twentyseven])
    dot_347 = Dot(normalize = True, axes = -1)([pooling_twentyone, pooling_twentyeight])
    dot_349 = Dot(normalize = True, axes = -1)([pooling_twentytwo, pooling_twentythree])
    dot_350 = Dot(normalize = True, axes = -1)([pooling_twentytwo, pooling_twentyfour])
    dot_351 = Dot(normalize = True, axes = -1)([pooling_twentytwo, pooling_twentyfive])
    dot_352 = Dot(normalize = True, axes = -1)([pooling_twentytwo, pooling_twentysix])
    dot_353 = Dot(normalize = True, axes = -1)([pooling_twentytwo, pooling_twentyseven])
    dot_354 = Dot(normalize = True, axes = -1)([pooling_twentytwo, pooling_twentyeight])
    dot_356 = Dot(normalize = True, axes = -1)([pooling_twentythree, pooling_twentyfour])
    dot_357 = Dot(normalize = True, axes = -1)([pooling_twentythree, pooling_twentyfive])
    dot_358 = Dot(normalize = True, axes = -1)([pooling_twentythree, pooling_twentysix])
    dot_359 = Dot(normalize = True, axes = -1)([pooling_twentythree, pooling_twentyseven])
    dot_360 = Dot(normalize = True, axes = -1)([pooling_twentythree, pooling_twentyeight])
    dot_361 = Dot(normalize = True, axes = -1)([pooling_twentyfour, pooling_twentyfive])
    dot_362 = Dot(normalize = True, axes = -1)([pooling_twentyfour, pooling_twentysix])
    dot_363 = Dot(normalize = True, axes = -1)([pooling_twentyfour, pooling_twentyseven])
    dot_364 = Dot(normalize = True, axes = -1)([pooling_twentyfour, pooling_twentyeight])
    dot_365 = Dot(normalize = True, axes = -1)([pooling_twentyfive, pooling_twentysix])
    dot_366 = Dot(normalize = True, axes = -1)([pooling_twentyfive, pooling_twentyseven])
    dot_367 = Dot(normalize = True, axes = -1)([pooling_twentyfive, pooling_twentyeight])
    dot_368 = Dot(normalize = True, axes = -1)([pooling_twentysix, pooling_twentyseven])
    dot_369 = Dot(normalize = True, axes = -1)([pooling_twentysix, pooling_twentyeight])
    dot_370 = Dot(normalize = True, axes = -1)([pooling_twentyseven, pooling_twentyeight])
    
    # Interaction gate
    sum_interaction = Add(name = "interaction_gate")([dot_1, dot_2, dot_3, dot_4, dot_5, dot_6, dot_7, dot_8, 
                                                     dot_9, dot_10, dot_11, dot_12, dot_13, dot_14, dot_15, dot_16,
                                                     dot_17, dot_18, dot_19, dot_20, dot_21, dot_22, dot_23, dot_24, 
                                                     dot_25, dot_26, dot_27, dot_28, dot_29, dot_30, dot_31, dot_32,
                                                     dot_33, dot_34, dot_35, dot_36, dot_37, dot_38, dot_39, dot_40, 
                                                     dot_41, dot_42, dot_43, dot_44, dot_45, dot_46, dot_47, dot_48,
                                                     dot_49, dot_50, dot_51, dot_52, dot_53, dot_54, dot_55, dot_56,
                                                     dot_57, dot_58, dot_59, dot_60, dot_61, dot_62, dot_63, dot_64,
                                                     dot_65, dot_66, dot_67, dot_68, dot_69, dot_70, dot_71, dot_72,
                                                     dot_73, dot_74, dot_75, dot_76, dot_77, dot_78, dot_79, dot_80,
                                                     dot_81, dot_82, dot_83, dot_84, dot_85, dot_86, dot_87, dot_88,
                                                     dot_89, dot_90, dot_91, dot_92, dot_93, dot_94, dot_95, dot_96, 
                                                     dot_97, dot_98, dot_99, dot_100, dot_101, dot_102, dot_103, dot_104,
                                                     dot_105, dot_106, dot_107, dot_108, dot_109, dot_110, dot_111, 
                                                     dot_112, dot_113, dot_114, dot_116, dot_117, dot_118, dot_119,
                                                     dot_120, dot_121, dot_122, dot_123, dot_124, dot_125, dot_126, dot_127,
                                                     dot_128, dot_129, dot_130, dot_131, dot_132, dot_133, dot_134, dot_135,
                                                     dot_136, dot_137, dot_138, dot_139, dot_140, dot_141, dot_142, dot_143, 
                                                     dot_144, dot_145, dot_146, dot_147, dot_148, dot_149, dot_150, dot_151, 
                                                     dot_152, dot_153, dot_154, dot_155, dot_156, dot_157, dot_158, dot_159,
                                                     dot_160, dot_161, dot_162, dot_163, dot_164, dot_165, dot_167,
                                                     dot_168, dot_169, dot_170, dot_171, dot_172, dot_173, dot_174, dot_175,
                                                     dot_176, dot_177, dot_178, dot_179, dot_180, dot_181, dot_182, dot_183, 
                                                     dot_184, dot_185, dot_187, dot_188, dot_189, dot_190, dot_191,
                                                     dot_192, dot_193, dot_194, dot_195, dot_196, dot_197, dot_198, dot_199,
                                                     dot_200, dot_201, dot_202, dot_203, dot_204, dot_206, dot_207,
                                                     dot_208, dot_209, dot_210, dot_211, dot_212, dot_213, dot_214, dot_215, 
                                                     dot_216, dot_217, dot_218, dot_219, dot_220, dot_221, dot_222, 
                                                     dot_224, dot_225, dot_226, dot_227, dot_228, dot_229, dot_230, dot_231,
                                                     dot_232, dot_233, dot_234, dot_235, dot_236, dot_237, dot_238, dot_239,
                                                     dot_241, dot_242, dot_243, dot_244, dot_245, dot_246, dot_247,
                                                     dot_248, dot_249, dot_250, dot_251, dot_252, dot_253, dot_254, dot_255,
                                                     dot_257, dot_258, dot_259, dot_260, dot_261, dot_262, dot_263,
                                                     dot_264, dot_265, dot_266, dot_267, dot_268, dot_269, dot_270, 
                                                     dot_272, dot_273, dot_274, dot_275, dot_276, dot_277, dot_278, dot_279,
                                                     dot_280, dot_281, dot_282, dot_283, dot_284, dot_286, dot_287,
                                                     dot_288, dot_289, dot_290, dot_291, dot_292, dot_293, dot_294, dot_295, 
                                                     dot_296, dot_297, dot_299, dot_300, dot_301, dot_302, dot_303,
                                                     dot_304, dot_305, dot_306, dot_307, dot_308, dot_309, dot_311,
                                                     dot_312, dot_313, dot_314, dot_315, dot_316, dot_317, dot_318, dot_319,
                                                     dot_320, dot_322, dot_323, dot_324, dot_325, dot_326, dot_327, 
                                                     dot_328, dot_329, dot_330, dot_332, dot_333, dot_334, dot_335,
                                                     dot_336, dot_337, dot_338, dot_339, dot_341, dot_342, dot_343,
                                                     dot_344, dot_345, dot_346, dot_347, dot_349, dot_350, dot_351,
                                                     dot_352, dot_353, dot_354, dot_356, dot_357, dot_358, dot_359, 
                                                     dot_360, dot_361, dot_362, dot_363, dot_364, dot_365, dot_366, dot_367, 
                                                     dot_368, dot_369, dot_370, dot_371, dot_372, dot_373, dot_374, dot_375,
                                                     dot_376, dot_377, dot_378, dot_379, dot_380, dot_381, dot_382, dot_383,
                                                     dot_384, dot_385, dot_386, dot_387, dot_388, dot_389, dot_391,
                                                     dot_392, dot_393, dot_394, dot_395])
                                                    
    
    # Fully connected layer 
    merged = Dense(1, activation = "sigmoid")(sum_interaction)
    model = Model(inputs = [city, country, hotel, rating, popularity,
                           locality, price, landmark], outputs = merged)
    model.compile(optimizer = "Adadelta", loss = "binary_crossentropy", metrics = ["accuracy"])
    
    return model

# Instantiate model and show parameters
model = hotel_embeddings()
model.summary()


# In[19]:


# Train Model

n_positive = 200

# Create generator 
generator = generate_batch(hotel_tuples, n_positive, negative_ratio = 1)

# Train

item_property = model.fit_generator(generator, epochs = 20, 
                                    steps_per_epoch = len(hotel_tuples) // n_positive, verbose = 2)


# In[ ]:


# Save model
model.save("../models/nn_scraped_hotels.h5")
model.save_weights("../models/nn_scraped_hotels_weights.h5")

