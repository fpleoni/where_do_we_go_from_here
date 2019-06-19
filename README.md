# Where do we go from here?

Working with Trivago.com and Hotels.com data to create a destination recommender system.


## Repo Index

Below is a list that will help you navigate the project's repository.

  ../jupyter_notebooks/ - **Notebooks with all the project stages**
  
  ../src_py/ - **Scripts for Neural Networks**
  
  ../wdg_service/ - **Scripts for web app API**
  
  ../data/ - **.csv files with all datasets, both scraped and downloaded**
  
  ../models/ - **.h5 files with model architecture and weights for all neural networks trained**
  
  ../images/ - **Images used for README.md**
  
  ../project_gear/ - **Presentation and Science Fair ready Jupyter Notebook**

## The Motivation

Recommender systems are an essential tool to enhance user experience. In the travel and hospitality industry, recommender systems play a crucial role in helping the user navigate through different offerings; effectively operating as a line of defense against consumer over-choice.

The relevancy of the recommendations is paramount to maintain user engagement and to improve conversion rates.

There are many hotel and traveling recommendation systems out there: Trivago.com, Hotels.com, Expedia.com, Booking.com; and this cater to a specific audience: Those who are planning a trip or vacation somewhere.

But who caters to those of us with wanderlust? The ones without plans written in stone, those in search of the unknown. 


## The Question

Can we provide valuable destination recommendations using neural network embeddings to represent hotel features, and calculating cosine similarity between embeddings?

## The Project

Using customer online interaction data from Trivago.com, and hotel data from Hotel.com, we created a recommender system that provides a relevant recommendation travel destination to the user.

In order to make the recommendations, we calculate cosine similarity between the embeddings of different features. These embeddings were obtained using a deep neural network with a customized architecture. The features are: City, Country, Hotel Name, Popularity, Rating, Price, Locality and Landmark.

Before deciding on using embeddings and cosine similarity to build the recommender system, we tried more common approaches, such as collaborative filtering and content based recommenders, none of these were were fruitful because of matrix sparsity and the curse of dimensionality: our matrix was 99.99% sparse, and the maximum sparsity acceptable to use either collaborative filter or content base recommenders is 99.5% - according to industry guidelines.

Finally, we created a prototype product (a web application) to showcase some of the functionalities developed for the recommender system.

## The Data

The dataset contains 422.380 hotels, in 22.567 different cities, across 198 countries.

The data was scraped from Hotel.com, and it also includes locality, landmark, popularity, rating, price, URL, and address.



We also had a second dataset from Trivago.com. This dataset was downloaded through the RecSys Challenge 2019 website, and it had information on 15 million user interactions in Trivago's website, recorded over a period of week.

## Exploratory Data Analysis

We explored both datasets in search of interesting and relevant pieces of information that could help steer the direction of the project, or that could be considered fun facts.




## Baseline Model

The baseline model was meant to tackle the cold start problem by offering unpersonalized recommendations.

These recommendations were obtained following these steps:

1. From the Trivago.com dataset get a list of the top 100 most popular cities with the highest click-through rate.

2. Filter the Hotels.com dataset so that only entries with the top 100 most popular cities remain.

3. Filter the results from **Step 2** so that only entries with a rating score of greater than or equal to 4.0/5.0 remain.

4. Filter the results from **Step 3** so that only entries with a popularity score of greater than or equal to 100 remain.

5. Sort the result from **Step 4** by *Popularity*, *Price*, and *Rating*. 

In the prototype web application, the baseline model is used to make recommendations in the home page.

## Final Model

The final model is a deep neural network with 450 layers. The main block-wise architecture is as following:

1. Input Layer

2. Embedding Layer

3. Interaction Gate Layer: Embedding Wise Multiplication

4. Global Max Pooling Layer

5. Interaction Gate Layer: Pool Wise Dot Product

6. Interaction Gate Layer: Sum

7. Dense Layer: Sigmoid Activation Function

The neural network was use to calculate the embeddings of all the features that defined a hotel entity in our dataset: Hotel Name, City, Country, Rating, Popularity, Price, Landmark and Locality. In order to achieve this we created a supervised learning classification problem, in which we had positive examples of hotels (real combinations of attributes) and negative examples of hotels (random -non real- combinations attributes), and trained the neural network to recognize one from the other; achieving a 90% accuracy score.

Other important parameters tuned for this task were the optimizer: **Adadelta**, and the loss function: **Binary Cross-entropy**.


## Next Steps

1. Use Multiplicative Long-Short Term Memory networks to create sequence-based recommendations using the Trivago.com dataset
2. Double the amount of hotels in the dataset
3. Create richer embeddings for hotel entities with reviews from Tripadvisor.com 
4. Create richer embeddings for hotel entities with hotel images from Hotels.com 
5. Perform cluster analysis on hotel embeddings
6. Broaden the functionalities of the prototype web application so that it mimics all the functionality developed for the recommender system
