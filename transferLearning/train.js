const { Classifier } = require('../mobilenetClassifier');

(async () => {
  await Classifier.init();

  await Classifier.addExamplesPath('trainset/grayscale/gs_0', 0);
  await Classifier.addExamplesPath('trainset/grayscale/gs_1', 1);

  console.log('palm3D_D2048_B01_Re-4_E50');
  await Classifier.train({
    denseUnits: 2048,
    batchSizeFraction: 0.1,
    learningRate: 0.00001,
    epochs: 50
  });
  await Classifier.saveModel('palm3D_D2048_B01_Re-4_E50');
})();
