const { Classifier } = require('../mobilenetClassifier');

(async () => {
  await Classifier.init();

  await Classifier.addExamplesPath('trainset/grayscale/gs_0', 0);
  await Classifier.addExamplesPath('trainset/grayscale/gs_1', 1);

  console.log('palmV3_D128_B01_Re-4_E50');
  await Classifier.train({
    denseUnits: 128,
    batchSizeFraction: 0.1,
    learningRate: 0.0001,
    epochs: 50
  });
  await Classifier.saveModel('palmV3_D128_B01_Re-4_E50');
})();
