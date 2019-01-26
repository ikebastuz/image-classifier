const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

const { ControllerDataset } = require('./controller_dataset');
const Utils = require('../utils');

// The number of classes we want to predict. In this example, we will be
const NUM_CLASSES = 2;
const IMG_SIZE = { width: 224, height: 224 };

// The dataset object where we will store activations.
const controllerDataset = new ControllerDataset(NUM_CLASSES);

let truncatedMobileNet;
let model;

// Loads mobilenet and returns a model that returns the internal activation
// we'll use as input to our classifier model.
async function loadTruncatedMobileNet() {
  const mobileNetModelLocal =
    'file://./../models/nnModels/mobileNet/mobilenet.json';
  const mobileNetModelHttp =
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';
  const mobilenet = await tf.loadModel(mobileNetModelLocal);

  // Return a model that outputs an internal activation.
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
}

// When the UI buttons are pressed, read a frame from the webcam and associate
// it with the class label given by the button. up, down, left, right are
// labels 0, 1, 2, 3 respectively.

function addExample(imageData, label) {
  tf.tidy(() => {
    const img = imageData;
    controllerDataset.addExample(truncatedMobileNet.predict(img), label);
  });
}

/**
 * Sets up and trains the classifier.
 */
async function train() {
  const epochs = 30;
  const denseUnits = 1024;
  const learningRate = 0.00001;
  const batchSizeFraction = 0.1;
  if (controllerDataset.xs == null) {
    throw new Error('Add some examples before training!');
  }

  // Creates a 2-layer fully connected model. By creating a separate model,
  // rather than adding layers to the mobilenet model, we "freeze" the weights
  // of the mobilenet model, and only train weights from the new model.
  model = tf.sequential({
    layers: [
      // Flattens the input to a vector so we can use it in a dense layer. While
      // technically a layer, this only performs a reshape (and has no training
      // parameters).
      tf.layers.flatten({
        inputShape: truncatedMobileNet.outputs[0].shape.slice(1)
      }),
      tf.layers.dense({
        units: denseUnits,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),
      tf.layers.dense({
        units: denseUnits / 2,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),
      tf.layers.dense({
        units: NUM_CLASSES,
        kernelInitializer: 'varianceScaling',
        useBias: false,
        activation: 'softmax'
      })
    ]
  });

  // Creates the optimizers which drives training of the model.
  const optimizer = tf.train.adam(learningRate);
  // We use categoricalCrossentropy which is the loss function we use for
  // categorical classification which measures the error between our predicted
  // probability distribution over classes (probability that an input is of each
  // class), versus the label (100% probability in the true class)>
  model.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy' });

  // We parameterize batch size as a fraction of the entire dataset because the
  // number of examples that are collected depends on how many examples the user
  // collects. This allows us to have a flexible batch size.
  const batchSize = Math.floor(
    controllerDataset.xs.shape[0] * batchSizeFraction
  );
  if (!(batchSize > 0)) {
    throw new Error(
      `Batch size is 0 or NaN. Please choose a non-zero fraction.`
    );
  }

  // Train the model! Model.fit() will shuffle xs & ys so we don't have to.

  await model.fit(controllerDataset.xs, controllerDataset.ys, {
    batchSize,
    epochs,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        console.log('Loss: ' + logs.loss.toFixed(5));
      }
    }
  });
}

async function predict(croppedImage) {
  const predictedClass = tf.tidy(() => {
    // Make a prediction through mobilenet, getting the internal activation of
    // the mobilenet model, i.e., "embeddings" of the input images.
    const embeddings = truncatedMobileNet.predict(croppedImage);

    // Make a prediction through our newly-trained model using the embeddings
    // from mobilenet as input.
    const predictions = model.predict(embeddings);

    // Returns the index with the maximum probability. This number corresponds
    // to the class the model thinks is the most probable given the input.
    return predictions.as1D().argMax();
  });

  const classId = (await predictedClass.data())[0];
  predictedClass.dispose();

  console.log(`Predicted class = ${classId}`);
  return classId;
}

async function trainModel(modelName) {
  await addExamplesPath('../trainset/sum_0', 0);
  await addExamplesPath('../trainset/sum_0_contr', 0);
  await addExamplesPath('../trainset/sum_0_v2', 0);

  await addExamplesPath('../trainset/sum_1', 1);
  await addExamplesPath('../trainset/sum_1_contr', 1);
  await addExamplesPath('../trainset/sum_1_v2', 1);

  await train();

  model.save(`file://./../models/nnModels/${modelName}`);
}

async function addExamplesPath(path, idx) {
  console.log(`Training idx: ${idx} from ${path}`);
  const fileNames = await Utils.getFileNames(path);
  for (let i = 0; i < fileNames.length; i++) {
    const canvas = await Utils.processImage(fileNames[i], IMG_SIZE);
    const croppedImage = await Utils.normalizeImage(canvas);
    addExample(croppedImage, idx);
  }
}

async function predictPath(path) {
  console.log(`Testing ${path}`);
  const fileNames = await Utils.getFileNames(path);

  const stats = {};
  for (let i = 0; i < fileNames.length; i++) {
    const canvas = await Utils.processImage(fileNames[i], IMG_SIZE);
    const croppedImage = await Utils.normalizeImage(canvas);
    const classIndex = await predict(croppedImage);
    if (!stats[classIndex]) stats[classIndex] = 0;

    stats[classIndex]++;
  }

  console.log(stats);
}

async function testModel(modelName) {
  model = await tf.loadModel(
    `file://./../models/nnModels/${modelName}/model.json`
  );

  await predictPath('../testset/0');
  await predictPath('../testset/1');
}

async function init() {
  truncatedMobileNet = await loadTruncatedMobileNet();
  console.log('MOBILENET LOADED!');

  //await trainModel('palm1024softmax000001');
  await testModel('palm1024softmax000001');
}

init();
