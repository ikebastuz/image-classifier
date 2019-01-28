const { Classifier } = require('../mobilenetClassifier');

(async () => {
  await Classifier.init();

  await Classifier.addExamplesPath('trainset/t0', 0);
  await Classifier.addExamplesPath('trainset/t1', 1);

  await Classifier.train({
    epochs: 5,
    denseUnits: 128,
    learningRate: 0.0001,
    batchSizeFraction: 0.5
  });

  await Classifier.saveModel('trainModel');
})();
