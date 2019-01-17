const express = require('express');
const bodyParser = require('body-parser');
const imgClassifier = require('./nnClassifier');
const { addImageToDataset } = require('./nnClassifier/utils');

const app = express();

const port = 8085;

app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());

app.use(function(req, res, next) {
  res.header('Access-Control-Allow-Origin', '*');
  res.header(
    'Access-Control-Allow-Headers',
    'Origin, X-Requested-With, Content-Type, Accept'
  );
  next();
});

/*  
  @route    POST /loadModel
  @desc     Load classifier into memory.
  @params   
    modelName: string (name of trained model. If empty - creates new classifier for training)
*/

app.post('/loadModel', async function(req, res) {
  try {
    const classifierName = req.body.modelName ? req.body.modelName : false;
    const result = await imgClassifier.initClassifier(classifierName);

    console.log(result);
    res.send({ data: result });
  } catch (e) {
    res.send({ data: e, error: true });
  }
});

/*  
  @route    POST /predict
  @desc     Classifies image and returns classIndex
  @params   
    imageData: string (image path or base64 image string)
*/

app.post('/predict', async function(req, res) {
  try {
    if (req.body.imageData) {
      const result = await imgClassifier.predictImage(req.body.imageData);

      console.log(result);
      res.send(result);
    }
  } catch (error) {
    res.send({ data: error, error: true });
  }
});

/*  
  @route    POST /trainBatch
  @desc     Add batch of image files to model
  @params   
    imageBatch: Object (path to image files)
      Object {
        path: string (path to image files),
        class: string/number (class Index for passed image)
      }
*/

app.post('/trainBatch', async function(req, res) {
  if (req.body.imageBatch) {
    try {
      const result = await imgClassifier.trainBatch(
        JSON.parse(req.body.imageBatch)
      );

      console.log(result);
      res.send(result);
    } catch (error) {
      res.send({ data: error, error: true });
    }
  }
});

/*  
  @route    POST /trainImage
  @desc     Add a single to model
  @params   
    imageData: Object (path to image file or base64 image string)
      Object {
        data: path to image file or base64 image string
        class: string/number (correct class of passed image)
      }
*/

app.post('/trainImage', async function(req, res) {
  if (req.body.imageData) {
    try {
      await addImageToDataset(req.body.imageData);
      const result = await imgClassifier.trainImage(req.body.imageData);

      console.log(result);
      res.send(result);
    } catch (e) {
      res.send(e);
    }
  }
});

/*  
  @route    POST /saveModel
  @desc     Save trained model
  @params   
    modelName: string (name of model to save)
*/

app.post('/saveModel', async function(req, res) {
  if (req.body.modelName) {
    try {
      const result = await imgClassifier.saveModel(req.body.modelName);

      console.log(result);
      res.send(result);
    } catch (e) {
      res.send(e);
    }
  }
});

/*  
  @route    POST /testModel
  @desc     Run test on loaded model
  @params   
    imageBatch: JSON array of objects (name of model to save)
      Object {
        path: string (Path to test set of images),
        class: string/number (correct class of passed images)
      }
*/

app.post('/testModel', async function(req, res) {
  if (req.body.imageBatch) {
    try {
      const imageFolders = JSON.parse(req.body.imageBatch);
      if (Array.isArray(imageFolders)) {
        const result = await imgClassifier.testModel(imageFolders);

        console.log(result);
        res.send(result);
      } else {
        res.send({ data: 'imageBatch is not an array' });
      }
    } catch (e) {
      res.send({ data: 'imageBatch is not a JSON' });
    }
  }
});

app.listen(port, function() {
  console.log(`Server running at http://127.0.0.1:${port}/`);
});
