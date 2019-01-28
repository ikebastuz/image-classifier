const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

const { ControllerDataset } = require('./controllerDataset');
const Utils = require('../utils');

class Classifier {
  constructor() {
    this.classifier = false;
    this.mobilenet = false;
    this.controllerDataset = false;
    this.IMAGE_SIZE = {
      width: 224,
      height: 224
    };
    this.NUM_CLASSES = 2;
  }
  async init() {
    this.controllerDataset = new ControllerDataset(this.NUM_CLASSES);
    console.log('dataset created');
    await this.loadMobileNet();
    console.log('mobilenet loaded');
  }

  async loadMobileNet() {
    const mobileNetModelLocal =
      'file://models/nnModels/mobileNet/mobilenet.json';
    const mobilenet = await tf.loadModel(mobileNetModelLocal);
    const layer = mobilenet.getLayer('conv_pw_13_relu');
    this.mobilenet = tf.model({
      inputs: mobilenet.inputs,
      outputs: layer.output
    });
  }

  async loadModel(modelName) {
    if (this.classifier) {
      this.classifier.dispose();
      this.classifier = false;
    }
    this.classifier = await tf.loadModel(
      `file://models/nnModels/${modelName}/model.json`
    );
    console.log(`${modelName} loaded`);
  }

  async addExamplesPath(path, idx) {
    console.log(`Adding examples for idx: ${idx} from ${path}`);
    const fileNames = await Utils.getFileNames(path);
    for (let i = 0; i < fileNames.length; i++) {
      const canvas = await Utils.processImage(fileNames[i], this.IMAGE_SIZE);
      const normImg = await Utils.normalizeImage(canvas);
      this.addExample(normImg, idx);
    }
  }

  addExample(img, idx) {
    tf.tidy(() => {
      this.controllerDataset.addExample(this.mobilenet.predict(img), idx);
    });
  }

  async train(config) {
    const { epochs, denseUnits, learningRate, batchSizeFraction } = config;

    if (this.controllerDataset.xs == null) {
      throw new Error('Add some examples before training!');
    }

    // Creates a 2-layer fully connected model. By creating a separate model,
    // rather than adding layers to the mobilenet model, we "freeze" the weights
    // of the mobilenet model, and only train weights from the new model.
    if (this.classifier) this.classifier.dispose();

    this.classifier = tf.sequential({
      layers: [
        // Flattens the input to a vector so we can use it in a dense layer. While
        // technically a layer, this only performs a reshape (and has no training
        // parameters).
        tf.layers.flatten({
          inputShape: this.mobilenet.outputs[0].shape.slice(1)
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
          units: this.NUM_CLASSES,
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
    this.classifier.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy'
    });

    // We parameterize batch size as a fraction of the entire dataset because the
    // number of examples that are collected depends on how many examples the user
    // collects. This allows us to have a flexible batch size.
    const batchSize = Math.floor(
      this.controllerDataset.xs.shape[0] * batchSizeFraction
    );

    if (!(batchSize > 0)) {
      throw new Error(
        `Batch size is 0 or NaN. Please choose a non-zero fraction.`
      );
    }

    // Train the model! Model.fit() will shuffle xs & ys so we don't have to.

    await this.classifier.fit(
      this.controllerDataset.xs,
      this.controllerDataset.ys,
      {
        batchSize,
        epochs,
        callbacks: {
          onBatchEnd: async (batch, logs) => {
            console.log('Loss: ' + logs.loss.toFixed(5));
          }
        }
      }
    );
  }

  async saveModel(modelName) {
    await this.classifier.save(`file://models/nnModels/${modelName}`);
    console.log(`${modelName} successfully saved`);
  }

  async testPath(path, expectedIdx) {
    console.log(`Testing ${path}. Expecting idx: ${expectedIdx}`);
    const fileNames = await Utils.getFileNames(path);

    const stats = { correct: 0, wrong: 0 };
    for (let i = 0; i < fileNames.length; i++) {
      const canvas = await Utils.processImage(fileNames[i], this.IMAGE_SIZE);
      const normImg = await Utils.normalizeImage(canvas);
      const predictedIdx = await this.predict(normImg);
      if (predictedIdx == expectedIdx) {
        stats.correct++;
      } else {
        stats.wrong++;
      }
    }
    stats['accuracy'] = `${(
      (stats.correct / (stats.correct + stats.wrong)) *
      100
    ).toFixed(2)}%`;
    console.log(stats);
  }

  async predict(img) {
    const predictedClass = tf.tidy(() => {
      // Make a prediction through mobilenet, getting the internal activation of
      // the mobilenet model, i.e., "embeddings" of the input images.
      const embeddings = this.mobilenet.predict(img);

      // Make a prediction through our newly-trained model using the embeddings
      // from mobilenet as input.
      const predictions = this.classifier.predict(embeddings);

      // Returns the index with the maximum probability. This number corresponds
      // to the class the model thinks is the most probable given the input.
      return predictions.as1D().argMax();
    });

    const classId = (await predictedClass.data())[0];
    predictedClass.dispose();
    return classId;
  }
}

module.exports = {
  Classifier: new Classifier()
};
