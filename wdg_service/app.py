from flask import Flask, jsonify, request
import data
import recommender
app = Flask(__name__)

city_weights = recommender.get_embeddings("city_embedding")
hotel_weights = recommender.get_embeddings("hotel_embedding")
city_country_list = recommender.get_all_cities()

@app.route("/cities")
def cities():
  return jsonify(city_country_list)

@app.route("/city", methods=["POST"])
def city():
  requested_city = request.get_json()["name"]

  recommended_cities = recommender.find_similar_cities(requested_city, city_weights, index_name = "city", n = 20, 
                        filtering = False, filter_name = None)
  recommended_hotels = recommender.citybased_recommendation_baseline(requested_city)

  return jsonify({"cities": recommended_cities, "hotels": recommended_hotels})

@app.route("/hotel", methods=["POST"])
def hotel():
  requested_hotel = request.get_json()["name"]
  
  recommended_hotels = recommender.find_similar_hotels(requested_hotel, hotel_weights, index_name = "hotel_name", n = 20, 
                        filtering = False, filter_name = None)

  return jsonify(recommended_hotels)

@app.route("/hotels/global")
def baseline():
  return jsonify(recommender.baseline())

