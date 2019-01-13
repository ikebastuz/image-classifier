const { performance } = require("perf_hooks");

const tf = require("@tensorflow/tfjs");
require("@tensorflow/tfjs-node");
//require("@tensorflow/tfjs-node-gpu");
const mobilenetModule = require("@tensorflow-models/mobilenet");

const Utils = require("./utils");
const ClassifierModel = require("./model");

global.fetch = require("node-fetch");

class imgClassifier {
  constructor() {
    this.knn = false;
    this.mobilenet = false;
    this.IMAGE_SIZE = {
      width: 227,
      height: 227
    };
    this.TOPK = 10;

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
    }
    if (!classifierName) {
      this.knn = await ClassifierModel.createEmptyClassifier();
    } else {
      this.knn = await ClassifierModel.loadClassifier(classifierName);
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

    const accur = Utils.getModelAccuracy(this.stats);

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

  async trainBatch(imageBatch) {
    if (!this.knn || !this.mobilenet)
      return { data: "Models not loaded", error: true };

    const trainStart = performance.now();

    await this.trainPath(imageBatch.path, imageBatch.class);

    const exampleCount = this.knn.getClassExampleCount();

    const trainEnd = performance.now();
    const trainTime = `${((trainEnd - trainStart) / 1000).toFixed(2)} secs`;

    return {
      data: {
        msg: `Successfully trained`,
        exampleCount,
        trainTime
      }
    };
  }

  async trainImage(imageData) {
    if (!this.knn || !this.mobilenet)
      return { data: "Models not loaded", error: true };

    await this.addImageToClass(imageData.data, imageData.class);

    const exampleCount = this.knn.getClassExampleCount();

    return {
      data: {
        msg: `Successfully trained`,
        exampleCount
      }
    };
  }

  async saveModel(modelName) {
    const numClasses = Number(this.knn.getNumClasses());
    if (numClasses < 2) {
      return {
        data: `Need to train at least two classes. Currentry trained ${numClasses}`,
        error: true
      };
    } else {
      console.log(`Saving model ${modelName}`);
      await ClassifierModel.saveClassifier(modelName, this.knn);
      return {
        data: {
          msg: `Successfully saved to ${modelName}`
        }
      };
    }
  }

  async trainPath(path, idx) {
    const imageNames = await Utils.getFileNames(path);
    for (let i = 0; i < imageNames.length; i++) {
      await this.addImageToClass(imageNames[i], idx);
    }
  }

  async addImageToClass(imageData, idx) {
    const img = await Utils.processImage(imageData, this.IMAGE_SIZE);
    const imgTf = tf.fromPixels(img);
    const inferLocal = img => this.mobilenet.infer(img, "conv_preds");
    const logits = inferLocal(imgTf);
    this.knn.addExample(logits, Number(idx));

    imgTf.dispose();
    if (logits != null) {
      logits.dispose();
    }
  }

  async predictPath(path, idx) {
    if (!this.knn || !this.mobilenet)
      return { data: "Models not loaded", error: true };

    const imageNames = await Utils.getFileNames(path);
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
    const img = await Utils.processImage(imageData, this.IMAGE_SIZE);
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
}

module.exports = new imgClassifier();
