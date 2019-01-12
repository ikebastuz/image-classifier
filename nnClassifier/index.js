const fs = require("fs");
const { resolve } = require("path");
const util = require("util");
const { performance } = require("perf_hooks");

const { createCanvas, Image } = require("canvas");

const tf = require("@tensorflow/tfjs");
require("@tensorflow/tfjs-node");
//require("@tensorflow/tfjs-node-gpu");
const mobilenetModule = require("@tensorflow-models/mobilenet");
const knnClassifier = require("@tensorflow-models/knn-classifier");

global.fetch = require("node-fetch");
const readDir = util.promisify(fs.readdir);
const stat = util.promisify(fs.stat);

class imgClassifier {
  constructor() {
    this.knn = false;
    this.mobilenet = false;
    this.IMAGE_SIZE = 227;
    this.TOPK = 10;
    this.layersGroups = [];
    this.loadingClassifier = [];
    this.classifier = knnClassifier.create();

    this.stats = {};

    this.initModel();
  }

  async initModel() {
    if (this.mobilenet) this.mobilenet.dispose();
    const mobileModelPath = "models/mobileNet/mobilenet.json";
    this.mobilenet = new mobilenetModule.MobileNet(1, 1);
    this.mobilenet.mobileModelPath = `file://${mobileModelPath}`;
    await this.mobilenet.load();
  }

  async initClassifier(classifierName = false) {
    if (this.knn) {
      this.knn.dispose();
      this.knn = null;
      this.layersGroups = [];
      this.loadingClassifier = [];
    }
    if (!classifierName) {
      this.knn = await knnClassifier.create();
    } else {
      this.knn = await this.loadClassifier(classifierName);
    }

    return classifierName
      ? `classifier ${classifierName} loaded`
      : "classifier initialized";
  }

  async testModel(imageBatch) {
    if (!this.knn || !this.mobilenet)
      return { data: "Models not loaded", error: true };

    this.stats = {};
    const testStart = performance.now();

    for (let i = 0; i < imageBatch.length; i++) {
      console.log(`Testing dataset ${imageBatch[i].class}`);
      await this.predictPath(imageBatch[i].path, imageBatch[i].class);
    }

    const accur = Object.values(this.stats).reduce(
      (acc, curr) => {
        const correctUpd = acc.correct + curr.correct;
        const errUpd = acc.error + curr.error;
        return {
          ...acc,
          correct: correctUpd,
          error: errUpd,
          accur: (correctUpd / (correctUpd + errUpd)) * 100
        };
      },
      { correct: 0, error: 0, accur: 0 }
    ).accur;

    const testEnd = performance.now();
    const testTime = `${((testEnd - testStart) / 1000).toFixed(2)} secs`;

    return {
      data: {
        testTime,
        accuracy: `${accur}%`,
        stats: this.stats
      }
    };
  }

  async trainModel(modelName, imageBatch) {
    if (!this.knn || !this.mobilenet)
      return { data: "Models not loaded", error: true };

    const trainStart = performance.now();

    for (let i = 0; i < imageBatch.length; i++) {
      console.log(`Training dataset ${imageBatch[i].class}...`);
      await this.trainPath(imageBatch[i].path, imageBatch[i].class);
      console.log(`Done`);
    }

    const exampleCount = this.knn.getClassExampleCount();
    console.log(`EXAMPLES:`);
    console.log(exampleCount);

    console.log(`Saving model ${modelName}`);
    await this.saveClassifier(modelName);

    const trainEnd = performance.now();
    const trainTime = `${((trainEnd - trainStart) / 1000).toFixed(2)} secs`;

    return {
      data: {
        msg: `Successfully trained and saved as ${modelName}`,
        trainTime
      }
    };
  }

  async getFileNames(path) {
    const subdirs = await readDir(path);
    const imageNames = await Promise.all(
      subdirs.map(async subdir => {
        const res = resolve(path, subdir);
        return (await stat(res)).isDirectory() ? getFiles(res) : res;
      })
    );

    return imageNames.filter(imageName =>
      imageName.match(/[a-zA-Z0-9\s_\\.\-\(\):]+(.jpg|.jpeg|.png)/i)
    );
  }

  async trainPath(path, idx) {
    const imageNames = await this.getFileNames(path);
    for (let i = 0; i < imageNames.length; i++) {
      const img = await this.processImage(`${imageNames[i]}`);
      const imgTf = tf.fromPixels(img);

      const inferLocal = img => this.mobilenet.infer(img, "conv_preds");
      const logits = inferLocal(imgTf);

      this.knn.addExample(logits, idx);

      imgTf.dispose();
      if (logits != null) {
        logits.dispose();
      }
    }
  }

  async predictPath(path, idx) {
    if (!this.knn || !this.mobilenet)
      return { data: "Models not loaded", error: true };

    const imageNames = await this.getFileNames(path);
    for (let i = 0; i < imageNames.length; i++) {
      const prediction = await this.predictImage(`${imageNames[i]}`);
      if (prediction.error) continue;
      if (!this.stats[idx]) this.stats[idx] = { correct: 0, error: 0 };
      if (prediction.data.classIndex == idx) {
        this.stats[idx]["correct"]++;
      } else {
        this.stats[idx]["error"]++;
      }
    }
  }

  async predictImage(imageData) {
    if (!this.knn || !this.mobilenet)
      return { data: "Models not loaded", error: true };
    const img = await this.processImage(imageData);
    const imgTf = tf.fromPixels(img);
    const inferLocal = () => this.mobilenet.infer(imgTf, "conv_preds");
    const logits = inferLocal();

    const prediction = await this.knn.predictClass(logits, this.TOPK);
    imgTf.dispose();
    if (logits != null) {
      logits.dispose();
    }
    return { data: prediction };
  }

  async processImage(imageData) {
    const canvas = createCanvas(this.IMAGE_SIZE, this.IMAGE_SIZE);
    const ctx = canvas.getContext("2d");
    const img = new Image();

    const promise = new Promise(resolve => {
      img.crossOrigin = "";
      img.onload = () => {
        const { x, y } = this.getOffsets(img);
        ctx.drawImage(img, x, y);
        resolve(canvas);
      };
    });

    img.src = `${imageData}`;
    return promise;
  }

  getOffsets(img) {
    let x, y;
    if (Number(img.width) > this.IMAGE_SIZE) {
      x = -(img.width - this.IMAGE_SIZE) / 2;
    } else {
      x = (this.IMAGE_SIZE - img.width) / 2;
    }

    if (Number(img.height) > this.IMAGE_SIZE) {
      y = -(img.height - this.IMAGE_SIZE) / 2;
    } else {
      y = (this.IMAGE_SIZE - img.height) / 2;
    }

    return { x, y };
  }

  async classifierSaveWrapper(passedClassifier) {
    let layersList = [];
    layersList[0] = []; // for the input layer name as a string
    layersList[1] = []; // for the input layer
    layersList[2] = []; // for the concatenate layer name as a string
    layersList[3] = []; // for the concatenate layer

    let maxClasses = passedClassifier.getNumClasses();

    for (let classIdx = 0; classIdx < maxClasses; classIdx++) {
      layersList[0][classIdx] = `classIdx_${classIdx}`; // input name as a string

      console.log(`define input for ${classIdx}`);
      layersList[1][classIdx] = tf.input({
        shape: passedClassifier.getClassifierDataset()[classIdx].shape[0],
        name: layersList[1][classIdx]
      });

      console.log("define dense for: " + classIdx);
      layersList[2][classIdx] = `classIdx_${classIdx}_Dense`;
      layersList[3][classIdx] = tf.layers
        .dense({ units: 1000, name: this.layersGroups[classIdx] })
        .apply(layersList[1][classIdx]);
    }

    console.log("Concatenate Paths");
    const concatLayer = tf.layers
      .concatenate({ axis: 1, name: "concatLayer" })
      .apply(layersList[3]);
    const concatLayerDense = tf.layers
      .dense({ units: 1, name: "concatLayerDense" })
      .apply(concatLayer);

    console.log("Define Model");
    const resultClassifierModel = tf.model({
      inputs: layersList[1],
      outputs: concatLayerDense
    });
    resultClassifierModel.summary();
    passedClassifier.getClassifierDataset()[0].print(true);

    for (let classIdx = 0; classIdx < maxClasses; classIdx++) {
      const myInWeight = await passedClassifier.getClassifierDataset()[
        classIdx
      ];
      resultClassifierModel.layers[classIdx + maxClasses].setWeights([
        myInWeight,
        tf.ones([1000])
      ]);
    }

    return resultClassifierModel;
  }

  async saveClassifier(modelName) {
    console.log("saving");
    console.log(modelName);
    const classifier = await this.classifierSaveWrapper(this.knn);
    console.log(classifier);
    classifier.save(`file://models/${modelName}`);
    classifier.summary(null, null, x => console.log(x));
    console.log("Trained model successfully saved");
  }

  async loadClassifier(modelName) {
    const loadedModel = await tf.loadModel(
      `file://models/${modelName}/model.json`
    );
    console.log(`loadedModel.layers.length : ${loadedModel.layers.length}`);

    const myMaxLayers = loadedModel.layers.length;
    const myDenseEnd = myMaxLayers - 2;
    const myDenseStart = myDenseEnd / 2;

    for (
      let myWeightLoop = myDenseStart;
      myWeightLoop < myDenseEnd;
      myWeightLoop++
    ) {
      this.loadingClassifier[myWeightLoop - myDenseStart] = loadedModel.layers[
        myWeightLoop
      ].getWeights()[0];
      this.layersGroups[myWeightLoop - myDenseStart] =
        loadedModel.layers[myWeightLoop].name;
    }
    /*
    console.log("Printing all the incoming classifiers");
    for (let x = 0; x < this.loadingClassifier.length; x++) {
      this.loadingClassifier[x].print(true);
    }
    */

    console.log("Activating Classifier");
    this.classifier.dispose();
    this.classifier.setClassifierDataset(this.loadingClassifier);
    console.log("Classifier loaded");
    return this.classifier;
  }
}

module.exports = new imgClassifier();
