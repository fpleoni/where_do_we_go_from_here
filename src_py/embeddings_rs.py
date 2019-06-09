
# coding: utf-8

# In[64]:


# Import libraries
from collections import Counter, OrderedDict
from itertools import chain
from keras.layers import Input, Embedding, Dot, Reshape, Dense
from keras.models import Model
import numpy as np
import pandas as pd
pd.set_option("display.max_rows", 500)
pd.options.display.max_colwidth = 1000
import random


# In[3]:


# Import csv file
properties_path = "../data/properties_and_cities.csv"
metadata_path = "../data/item_metadata.csv"
train_path = "../data/train.csv"


# In[4]:


# Create DataFrames
trivago_df = pd.read_csv(train_path)


# In[5]:


properties_df = pd.read_csv(properties_path, usecols = ["item_id", "properties", "city"])


# In[21]:


metadata_df = pd.read_csv(metadata_path)


# In[7]:


item_list = metadata_df["item_id"].tolist()


# In[42]:


# Create indexes for each item
item_index = {item: idx for idx, item in enumerate(item_list)}
index_item = {idx: item for item, idx in item_index.items()}


# In[43]:


item_list[:10]


# In[22]:


metadata_df["properties"] = metadata_df["properties"].str.split("|")


# In[9]:


metadata_df.head()


# In[23]:


# Convert all properties to lowercase
metadata_df["properties"] = metadata_df["properties"].apply(lambda x: [w.lower() for w in x])


# In[24]:


metadata_df.head()


# In[25]:


properties_list = metadata_df["properties"].tolist()


# In[44]:


properties_list[:10]


# In[30]:


# Find set of unique properties and convert to a list
unique_properties = list(chain(*[list(set(tags)) for tags in properties_list]))


# In[32]:


# Count unique properties
def count_items(l):
    
    # Create a counter object
    counts = Counter(l)
    
    # Sort by highest count first and place in ordered dictionary
    counts = sorted(counts.items(), key = lambda x: x[1], reverse = True)
    counts = OrderedDict(counts)
    
    return counts

properties_count = count_items(unique_properties)


# In[35]:


# Sanity check
list(properties_count.items())[:10]


# In[37]:


# Create indexes for each property
property_index = {tag: idx for idx, tag in enumerate(unique_properties)}
index_properties = {idx: tag for tag, idx in property_index.items()}


# In[45]:


# Build item properties pair to train embedding neural network
item_property_pairs = []

# Iterate through each row of dataframe
for index, row in metadata_df.iterrows():
    # Iterate through the properties in the item
    item_property_pairs.extend((item_index[metadata_df.at[index, "item_id"]], property_index[tag.lower()]) for tag in 
                               metadata_df.at[index, "properties"] if tag.lower() in unique_properties)


# In[46]:


# Sanity check
len(item_property_pairs)


# In[58]:


# Sanity check
index_item[item_property_pairs[0][0]], index_properties[item_property_pairs[0][1]]


# In[60]:


# Generator for training samples
def generate_batch(pairs, n_positive = 50, negative_ratio = 1.0):
    
    pairs_set = set(pairs)
    
    batch_size = n_positive * (1 + negative_ratio)
    batch = np.zeros((batch_size, 3))
    
    # Label for negative examples
    neg_label = 0
    
    # This creates a generator
    while True:
        # randomly choose positive examples
        for idx, (item_id, property_id) in enumerate(random.sample(pairs, n_positive)):
            batch[idx, :] = (item_id, property_id, 1)

        # Increment idx by 1
        idx += 1
        
        # Add negative examples until reach batch size
        while idx < batch_size:
            
            # random selection
            random_item = random.randrange(len(item_list))
            random_property = random.randrange(len(unique_properties))
            
            # Check to make sure this is not a positive example
            if (random_item, random_property) not in pairs_set:
                
                # Add to batch and increment index
                batch[idx, :] = (random_item, random_property, neg_label)
                idx += 1
                
        # Make sure to shuffle order
        np.random.shuffle(batch)
        yield {"item": batch[:, 0], "property": batch[:, 1]}, batch[:, 2]


# In[63]:


# Properties embedding model
def hotel_embeddings(embedding_size = 50):
    
    # Inputs are one-dimensional
    item = Input(name = "item", shape = [1])
    tag = Input(name = "property", shape = [1])
    
    # Embedding the item
    item_embedding = Embedding(name = "item_embedding", input_dim = len(item_index), 
                              output_dim = embedding_size)(item)
    
    # Embedding the properties
    property_embedding = Embedding(name = "property_embedding", input_dim = len(property_index),
                                  output_dim = embedding_size)(tag)
    
    # Merge the embeddings with dot product across second axis
    merged = Dot(name = "dot_product", normalize = True, axes = 2)([item_embedding, property_embedding])
    
    # Reshape to get a single number
    merged = Reshape(target_shape = [1])(merged)
    
    merged = Dense(1, activation = "sigmoid")(merged)
    model = Model(inputs = [item, tag], outputs = merged)
    model.compile(optimizer = "Adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    
    return model

# Instantiate model and show parameters
model = hotel_embeddings()
model.summary()


# In[ ]:


# Train Model

n_positive = 2000

# Create generator 
generator = generate_batch(item_property_pairs, n_positive, negative_ratio = 1)

# Train

item_property = model.fit_generator(generator, epochs = 100, 
                                    steps_per_epoch = len(item_property_pairs) // n_positive, verbose = 2)


# In[ ]:


# Save model
model.save("../models/embeddings_first_attempt.h5")
model.save_weights("../models/embeddings_first_attempt_weights.h5")


# In[ ]:


# Extract embeddings
hotel_layer = model.get_layer("item_embedding")
hotel_weights = hotel_layer.get_weights()[0]

# Normalize the embeddings so that we can calculate cosine similarity
hotel_weights = hotel_weights / np.linalg.norm(hotel_weights, axis = 1).reshape((-1, 1))

