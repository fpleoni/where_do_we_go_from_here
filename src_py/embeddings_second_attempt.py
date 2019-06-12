
# coding: utf-8

# In[1]:


# Import libraries
from collections import Counter, OrderedDict
from itertools import chain
from keras.layers import Input, Embedding, Dot, Reshape, Dense, Concatenate, Multiply
from keras.models import Model
import numpy as np
import pandas as pd
pd.set_option("display.max_rows", 500)
pd.options.display.max_colwidth = 1000
import random


# In[2]:


# Path to file
hotels_path = "../data/hotels_items.csv"


# In[3]:


# Create dataframe
hotels_df = pd.read_csv(hotels_path, usecols = ["item_id", "properties", "city", "price"])


# In[4]:


# check info
hotels_df.info()


# In[5]:


hotels_df.head()


# In[6]:


# Create item list
item_list = hotels_df["item_id"].tolist()

# Create indexes for each item
item_index = {item: idx for idx, item in enumerate(item_list)}
index_item = {idx: item for item, idx in item_index.items()}

# Create price list
price_list = hotels_df["price"].tolist()

# Create indexes for each item
price_index = {price: idx for idx, price in enumerate(price_list)}
index_price = {idx: price for price, idx in price_index.items()}


# In[7]:


# Prepare properties column

# Split of pipe
hotels_df["properties"] = hotels_df["properties"].str.split("|")

# Convert all properties to lowercase
hotels_df["properties"] = hotels_df["properties"].apply(lambda x: [w.lower() for w in x])

# Create list
properties_list = hotels_df["properties"].tolist()

# Find set of unique properties and convert to a list
unique_properties = list(chain(*[list(set(tags)) for tags in properties_list]))
unique_properties = set(unique_properties)

# Create indexes for each property
property_index = {tag: idx for idx, tag in enumerate(unique_properties)}
index_properties = {idx: tag for tag, idx in property_index.items()}


# In[9]:


# Prepare city column

# Convert all properties to lowercase
hotels_df["city"] = hotels_df["city"].apply(lambda x: x.lower())

# Create list
location_list = hotels_df["city"].tolist()

# Find set of unique locations and convert to a list
# unique_locations = list(chain(*[list(set(location)) for location in location_list]))
unique_locations = set(location_list)

# Create indexes for each property
location_index = {location: idx for idx, location in enumerate(unique_locations)}
index_location = {idx: location for location, idx in location_index.items()}


# In[10]:


# Build tuples to train embedding neural network
hotel_tuples = []

# Iterate through each row of dataframe
for index, row in hotels_df.iterrows():
    # Iterate through the properties in the item
    hotel_tuples.extend((item_index[hotels_df.at[index, "item_id"]], property_index[tag.lower()], 
                         price_index[hotels_df.at[index, "price"]], location_index[hotels_df.at[index, "city"]]) 
                        for tag in hotels_df.at[index, "properties"] if tag.lower() in unique_properties)


# In[11]:


# Generator for training samples
def generate_batch(tuples, n_positive = 75, negative_ratio = 2.0):
    
    pairs_set = set(tuples)
    
    batch_size = n_positive * (1 + negative_ratio)
    batch = np.zeros((batch_size, 5))
    
    # Label for negative examples
    neg_label = 0
    
    # This creates a generator
    while True:
        # randomly choose positive examples
        for idx, (item_id, property_id, price_id, location_id) in enumerate(random.sample(tuples, n_positive)):
            batch[idx, :] = (item_id, property_id, price_id, location_id, 1)

        # Increment idx by 1
        idx += 1
        
        # Add negative examples until reach batch size
        while idx < batch_size:
            
            # random selection
            random_item = random.randrange(len(item_list))
            random_property = random.randrange(len(unique_properties))
            random_price = random.randrange(len(price_list))
            random_location = random.randrange(len(unique_locations))
            
            # Check to make sure this is not a positive example
            if (random_item, random_property, random_price, random_location) not in pairs_set:
                
                # Add to batch and increment index
                batch[idx, :] = (random_item, random_property, random_price, random_location, neg_label)
                idx += 1
                
        # Make sure to shuffle order
        np.random.shuffle(batch)
        yield {"item": batch[:, 0], "property": batch[:, 1], "price": batch[:, 2], "location": batch[:, 3]}, batch[:, 4]


# In[12]:


# Properties embedding model
def hotel_embeddings(embedding_size = 100):
    
    # Inputs are one-dimensional
    item = Input(name = "item", shape = [1])
    tag = Input(name = "property", shape = [1])
    price = Input(name = "price", shape = [1])
    location = Input(name = "location", shape = [1])
    
    # Embedding the item
    item_embedding = Embedding(name = "item_embedding", input_dim = len(item_index), 
                              output_dim = embedding_size)(item)
    
    # Embedding the properties
    property_embedding = Embedding(name = "property_embedding", input_dim = len(property_index),
                                  output_dim = embedding_size)(tag)
    
    # Embedding the properties
    price_embedding = Embedding(name = "price_embedding", input_dim = len(price_index),
                                  output_dim = embedding_size)(price)
    
    # Embedding the properties
    location_embedding = Embedding(name = "location_embedding", input_dim = len(location_index),
                                  output_dim = embedding_size)(location)
    
    # Merge the embeddings with dot product across second axis
    merged_one = Dot(name = "dot_product_one", normalize = True, axes = 2)([item_embedding, property_embedding])
    merged_two = Dot(name = "dot_product_two", normalize = True, axes = 2)([item_embedding, price_embedding])
    merged_three = Dot(name = "dot_product_three", normalize = True, axes = 2)([item_embedding, location_embedding])
    merged_four = Dot(name = "dot_product_four", normalize = True, axes = 2)([price_embedding, property_embedding])
    merged_five = Dot(name = "dot_product_five", normalize = True, axes = 2)([location_embedding, property_embedding])
    merged_six = Dot(name = "dot_product_six", normalize = True, axes = 2)([location_embedding, price_embedding])
    
    # Concatenate all the merged dot products
    multiply = Multiply(name = "multiplication")([merged_one, merged_two, merged_three, merged_four, merged_five, merged_six])
    
    # Reshape to get a single number
    merged = Reshape(target_shape = [1])(multiply)
    
    merged = Dense(1, activation = "sigmoid")(merged)
    model = Model(inputs = [item, tag, price, location], outputs = merged)
    model.compile(optimizer = "Adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
    
    return model

# Instantiate model and show parameters
model = hotel_embeddings()
model.summary()


# In[ ]:


# Train Model

n_positive = 2000

# Create generator 
generator = generate_batch(hotel_tuples, n_positive, negative_ratio = 1)

# Train

item_property = model.fit_generator(generator, epochs = 100, 
                                    steps_per_epoch = len(hotel_tuples) // n_positive, verbose = 2)


# In[ ]:


# Save model
model.save("../models/embeddings_second_attempt.h5")
model.save_weights("../models/embeddings_second_attempt_weights.h5")

