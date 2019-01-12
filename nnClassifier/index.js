const fs = require("fs");
const { resolve } = require("path");
const util = require("util");

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
    this.IMAGE_SIZE = 227;
    this.TOPK = 10;
    this.myGroups = [];
    this.myIncomingClassifier = [];
    this.classifier = knnClassifier.create();

    this.stats = {};

    this.loadModel();
  }

  async loadModel(forTraining = false) {
    const mobileModelPath = "models/mobileNet/mobilenet.json";
    this.mobilenet = new mobilenetModule.MobileNet(1, 1);
    this.mobilenet.mobileModelPath = `file://${mobileModelPath}`;
    await this.mobilenet.load();

    if (forTraining) {
      this.knn = await knnClassifier.create();
      await this.trainData();
      this.saveClassifier(this.knn, "cat_dog_model");
    } else {
      this.knn = await this.loadClassifier(
        "models/catdogmodel_2000/model.json"
      );
      await this.predictData();
    }
  }

  async predictData() {
    console.log("PREDICTING Dataset 0");
    await this.predictPath("./dataset/test/a", "0");

    console.log("PREDICTING Dataset 1");
    await this.predictPath("./dataset/test/b", "1");

    console.log(this.stats);
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
    console.log(`Accuracy: ${accur}%`);
  }

  async trainData() {
    await this.trainPath("./dataset/train/a", 0);
    console.log("TRAINED A");
    await this.trainPath("./dataset/train/b", 1);
    console.log("TRAINED B");

    const exampleCount = this.knn.getClassExampleCount();
    console.log(`EXAMPLES:`);
    console.log(exampleCount);
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
    const imageNames = await this.getFileNames(path);
    for (let i = 0; i < imageNames.length; i++) {
      const prediction = await this.predictImage(`${imageNames[i]}`);
      if (!this.stats[idx]) this.stats[idx] = { correct: 0, error: 0 };
      if (prediction.classIndex == idx) {
        this.stats[idx]["correct"]++;
      } else {
        this.stats[idx]["error"]++;
      }
    }
  }

  async predictImage(imagePath) {
    const img = await this.processImage(imagePath);
    const imgTf = tf.fromPixels(img);
    const inferLocal = () => this.mobilenet.infer(imgTf, "conv_preds");
    const logits = inferLocal();
    const prediction = await this.knn.predictClass(logits, this.TOPK);
    imgTf.dispose();
    if (logits != null) {
      logits.dispose();
    }
    return prediction;
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

  async classifierSaveWrapper(myPassedClassifier) {
    let myLayerList = [];
    myLayerList[0] = []; // for the input layer name as a string
    myLayerList[1] = []; // for the input layer
    myLayerList[2] = []; // for the concatenate layer name as a string
    myLayerList[3] = []; // for the concatenate layer

    let myMaxClasses = myPassedClassifier.getNumClasses();

    for (
      let myClassifierLoop = 0;
      myClassifierLoop < myMaxClasses;
      myClassifierLoop++
    ) {
      myLayerList[0][myClassifierLoop] = `myInput_${myClassifierLoop}`; // input name as a string

      console.log(`define input for ${myClassifierLoop}`);
      myLayerList[1][myClassifierLoop] = tf.input({
        shape: myPassedClassifier.getClassifierDataset()[myClassifierLoop]
          .shape[0],
        name: myLayerList[1][myClassifierLoop]
      });

      console.log("define dense for: " + myClassifierLoop);
      myLayerList[2][myClassifierLoop] = `myInput_${myClassifierLoop}_Dense1`; // concatenate as a string
      myLayerList[3][myClassifierLoop] = tf.layers
        .dense({ units: 1000, name: this.myGroups[myClassifierLoop] })
        .apply(myLayerList[1][myClassifierLoop]); //Define concatenate layer
    }

    console.log("Concatenate Paths");
    const myConcatenate1 = tf.layers
      .concatenate({ axis: 1, name: "myConcatenate1" })
      .apply(myLayerList[3]); // send the entire list of dense
    const myConcatenate1Dense4 = tf.layers
      .dense({ units: 1, name: "myConcatenate1Dense4" })
      .apply(myConcatenate1);

    console.log("Define Model");
    const myClassifierModel = tf.model({
      inputs: myLayerList[1],
      outputs: myConcatenate1Dense4
    });
    myClassifierModel.summary();
    myPassedClassifier.getClassifierDataset()[0].print(true);

    for (
      let myClassifierLoop = 0;
      myClassifierLoop < myMaxClasses;
      myClassifierLoop++
    ) {
      const myInWeight = await myPassedClassifier.getClassifierDataset()[
        myClassifierLoop
      ];
      myClassifierModel.layers[myClassifierLoop + myMaxClasses].setWeights([
        myInWeight,
        tf.ones([1000])
      ]);
    }

    return myClassifierModel;
  }

  async saveClassifier(classifier, modelName) {
    const classifier = await this.classifierSaveWrapper(classifier); // pass global classifier
    classifier.save(`file://models/${modelName}`);
    classifier.summary(null, null, x => console.log(x));
    console.log("Trained model successfully saved");
  }

  async loadClassifier(modelPath) {
    const loadedModel = await tf.loadModel(`file://${modelPath}`);
    console.log(`loadedModel.layers.length : ${loadedModel.layers.length}`);

    const myMaxLayers = loadedModel.layers.length;
    const myDenseEnd = myMaxLayers - 2;
    const myDenseStart = myDenseEnd / 2; // assume 0 = first layer: if 6 layers 0-1 input, 2-3 dense, 4 concatenate, 5 dense output

    for (
      let myWeightLoop = myDenseStart;
      myWeightLoop < myDenseEnd;
      myWeightLoop++
    ) {
      console.log(
        `loadedModel.layers[${myWeightLoop}].getWeights()[0].print(true)`
      );

      this.myIncomingClassifier[
        myWeightLoop - myDenseStart
      ] = loadedModel.layers[myWeightLoop].getWeights()[0];
      this.myGroups[myWeightLoop - myDenseStart] =
        loadedModel.layers[myWeightLoop].name;
    }
    console.log("Printing all the incoming classifiers");
    for (let x = 0; x < this.myIncomingClassifier.length; x++) {
      this.myIncomingClassifier[x].print(true);
    }
    console.log("Activating Classifier");

    this.classifier.dispose();
    this.classifier.setClassifierDataset(this.myIncomingClassifier);
    console.log("Classifier loaded");
    return this.classifier;
  }
}

module.exports = new imgClassifier();
