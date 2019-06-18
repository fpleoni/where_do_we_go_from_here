from flask import Flask, jsonify, request
#import recommender
import random
import data
import recommender
app = Flask(__name__)


@app.route("/cities")
def cities():
  return jsonify(data.all_cities())

@app.route('/city', methods=['POST'])
def city():
  requested_city = request.get_json()["name"]
  
  city_weights = recommender.get_embeddings("city_embedding")
  recommended_cities = recommender.find_similar(requested_city, city_weights, index_name = "city", n = 20, 
                        filtering = False, filter_name = None)

  return jsonify(recommended_cities)

@app.route("/hotel", methods=['POST'])
def hotel():
  requested_hotel = request.get_json()["name"]
  
  hotel_weights = recommender.get_embeddings("hotel_embedding")
  recommended_hotels = recommender.find_similar(requested_hotel, hotel_weights, index_name = "hotel_name", n = 20, 
                        filtering = False, filter_name = None)

  return jsonify(recommended_hotels)

@app.route("/baseline")
def baseline():
  return jsonify(recommender.baseline())