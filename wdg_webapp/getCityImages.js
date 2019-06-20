const request = require('request');
var fs = require('fs');

const API_KEY = 'AIzaSyCyIBXuzgvRnhZoA78K88tYYKHXyc8_1CU'


var cities = ["Hotel Alvear, Argentina"]

var tasks = []
cities.forEach((city) => {
  tasks.push(() => {
    return new Promise((resolve) => {
      var properties = { input: city, inputtype: 'textquery', fields: "photos", key: API_KEY};
      request({url: 'https://maps.googleapis.com/maps/api/place/findplacefromtext/json', json:true, qs: properties}, (err, res, body) => {
          if (err) { return console.log(err); }
            console.log(body)
            var photoApiProperties = {
              photoreference: body.candidates[0].photos[0].photo_reference,
              maxwidth: 1800,
              key: API_KEY
            }
            request({url: 'https://maps.googleapis.com/maps/api/place/photo', qs: photoApiProperties}).pipe(fs.createWriteStream("./public/images/hotels/"+city.replace("/","")))

          resolve()
      });
    })
    
  })
})

function createQueue(tasks, maxNumOfWorkers = 4) {
  var numOfWorkers = 0;
  var taskIndex = 0;

  return new Promise(done => {
    const handleResult = index => result => {
      tasks[index] = result;
      numOfWorkers--;
      getNextTask();
    };
    const getNextTask = () => {
      console.log(taskIndex);
      if (numOfWorkers < maxNumOfWorkers && taskIndex < tasks.length) {
        tasks[taskIndex]().then(handleResult(taskIndex)).catch(handleResult(taskIndex));
        taskIndex++;
        numOfWorkers++;
        getNextTask();
      } else if (numOfWorkers === 0 && taskIndex === tasks.length) {
        done(tasks);
      }
    };
    getNextTask();
  });
}


createQueue(tasks).then(results => {
  console.log(results)
});
