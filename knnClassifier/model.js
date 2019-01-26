const tf = require('@tensorflow/tfjs');
const knnClassifier = require('@tensorflow-models/knn-classifier');

async function saveClassifier(modelName, knn) {
  const classifier = await classifierSaveWrapper(knn);
  classifier.save(`file://models/nnModels/${modelName}`);
  classifier.summary(null, null, (x) => console.log(x));
  console.log('Trained model successfully saved');
}

async function classifierSaveWrapper(passedClassifier) {
  let layersGroups = [];
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

    console.log('define dense for: ' + classIdx);
    layersList[2][classIdx] = `classIdx_${classIdx}_Dense`;
    layersList[3][classIdx] = tf.layers
      .dense({ units: 1000, name: layersGroups[classIdx] })
      .apply(layersList[1][classIdx]);
  }

  console.log('Concatenate Paths');
  const concatLayer = tf.layers
    .concatenate({ axis: 1, name: 'concatLayer' })
    .apply(layersList[3]);
  const concatLayerDense = tf.layers
    .dense({ units: 1, name: 'concatLayerDense' })
    .apply(concatLayer);

  console.log('Define Model');
  const resultClassifierModel = tf.model({
    inputs: layersList[1],
    outputs: concatLayerDense
  });
  resultClassifierModel.summary();
  passedClassifier.getClassifierDataset()[0].print(true);

  for (let classIdx = 0; classIdx < maxClasses; classIdx++) {
    const myInWeight = await passedClassifier.getClassifierDataset()[classIdx];
    resultClassifierModel.layers[classIdx + maxClasses].setWeights([
      myInWeight,
      tf.ones([1000])
    ]);
  }

  return resultClassifierModel;
}

async function loadClassifier(modelName) {
  const classifier = knnClassifier.create();
  const loadedModel = await tf.loadModel(
    `file://./models/nnModels/${modelName}/model.json`
  );
  console.log(`loadedModel.layers.length : ${loadedModel.layers.length}`);

  const myMaxLayers = loadedModel.layers.length;
  const myDenseEnd = myMaxLayers - 2;
  const myDenseStart = myDenseEnd / 2;

  const loadingClassifier = [],
    layersGroups = [];
  for (
    let myWeightLoop = myDenseStart;
    myWeightLoop < myDenseEnd;
    myWeightLoop++
  ) {
    loadingClassifier[myWeightLoop - myDenseStart] = loadedModel.layers[
      myWeightLoop
    ].getWeights()[0];
    layersGroups[myWeightLoop - myDenseStart] =
      loadedModel.layers[myWeightLoop].name;
  }

  /*
  console.log("Printing all the incoming classifiers");
  for (let x = 0; x < loadingClassifier.length; x++) {
    loadingClassifier[x].print(true);
  }
  */

  console.log('Activating Classifier');
  classifier.dispose();
  classifier.setClassifierDataset(loadingClassifier);
  console.log('Classifier loaded');
  return classifier;
}

async function createEmptyClassifier() {
  const classifier = await knnClassifier.create();
  return classifier;
}

module.exports = {
  saveClassifier,
  loadClassifier,
  createEmptyClassifier
};
