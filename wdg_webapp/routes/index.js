const express = require('express');
const request = require('request');

const router = express.Router();

router.get('/', function (req, res) {
    request.get({url: 'http://localhost:5000/hotels/global', json: true},
        (error, response, body) => {
        res.render('index', {
            title: 'Where do we go from here?',
            hotels: body
        })
    })
})

router.post('/city', function (req, res) {
    const city = req.body.city
    request.post({url: 'http://localhost:5000/city', body: {name: city}, json: true},
        (error, response, body) => {        
        res.render('city', {
            title: city,
            name: city,
            hotels: body.hotels,
            cities:  body.cities
        })
    })
})

router.post('/hotel', function (req, res) {
    const name = req.body.hotel
    const city = req.body.city
    request.post({url: 'http://localhost:5000/hotel', body: {name: name}, json: true},
        (error, response, body) => {
        res.render('hotel', {
            title: name + " (" + city + ")",
            hotels: body,
            city: city
        })
    })
})

module.exports = router;