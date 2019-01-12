const express = require("express");
const bodyParser = require("body-parser");
const imgClassifier = require("./nnClassifier");

const app = express();

const port = 8085;

app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());

app.use(function(req, res, next) {
  res.header("Access-Control-Allow-Origin", "*");
  res.header(
    "Access-Control-Allow-Headers",
    "Origin, X-Requested-With, Content-Type, Accept"
  );
  next();
});

app.post("/predict", async function(req, res) {
  let result = "Predicting image";
  try {
    result = await imgClassifier.predictImage(req.body.imageData);
  } catch (error) {
    result = error;
  }
  console.log(result);
  res.send({ msg: result.classIndex });
});

app.listen(port, function() {
  console.log(`Server running at http://127.0.0.1:${port}/`);
});
