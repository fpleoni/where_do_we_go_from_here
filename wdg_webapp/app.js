const express = require('express');
const bodyParser = require('body-parser');
const routes = require('./routes/index');

const app = express();

app.use(bodyParser.urlencoded({ extended: true }));
app.use('/static', express.static('public'))
app.use('/', routes);
app.set('views', './views')
app.set('view engine', 'pug')

const server = app.listen(3000, () => {
  console.log(`Express is running on port ${server.address().port}`);
});