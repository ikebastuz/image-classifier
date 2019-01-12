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

app.post("/loadModel", async function(req, res) {
  try {
    const classifierName = req.body.modelName ? req.body.modelName : false;
    const result = await imgClassifier.initClassifier(classifierName);
    res.send({ data: result });
  } catch (e) {
    res.send({ data: e, error: true });
  }
});

app.post("/predict", async function(req, res) {
  let result = "Recieved request";

  try {
    if (req.body.imageData) {
      // Predicting image
      result = await imgClassifier.predictImage(req.body.imageData);
      res.send(result);
    }
  } catch (error) {
    res.send({ data: error, error: true });
  }
});

app.post("/trainModel", async function(req, res) {
  if (req.body.imageBatch) {
    // training Model
    try {
      const imageFolders = JSON.parse(req.body.imageBatch);
      console.log(imageFolders);
      if (Array.isArray(imageFolders)) {
        console.log("HERE");
        result = await imgClassifier.trainModel(
          req.body.modelName,
          imageFolders
        );
        res.send(result);
      } else {
        res.send({ data: "imageBatch is not an array" });
      }
    } catch {
      res.send({ data: "imageBatch is not a JSON" });
    }
  }
});

app.post("/testModel", async function(req, res) {
  if (req.body.imageBatch) {
    // testing Model
    try {
      const imageFolders = JSON.parse(req.body.imageBatch);
      if (Array.isArray(imageFolders)) {
        result = await imgClassifier.testModel(imageFolders);
        res.send(result);
      } else {
        res.send({ data: "imageBatch is not an array" });
      }
    } catch {
      res.send({ data: "imageBatch is not a JSON" });
    }
  }
});

app.listen(port, function() {
  console.log(`Server running at http://127.0.0.1:${port}/`);
});
