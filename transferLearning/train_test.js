const { Classifier } = require('../mobilenetClassifier');

(async () => {
  const version = 7;
  const config = {
    denseUnits: 128,
    batchSizeFraction: 0.1,
    learningRate: 0.0001,
    epochs: 50
  };
  await Classifier.init();

  // Add train examples
  await Classifier.addExamplesPath('trainset/grayscale/train/gs_0', 0);
  await Classifier.addExamplesPath('trainset/grayscale/train/gs_1', 1);

  // Train
  await Classifier.train(config);

  // Test
  await Classifier.testPath('trainset/grayscale/val/gs_0', 0);
  await Classifier.testPath('trainset/grayscale/val/gs_1', 1);

  // Add test examples
  await Classifier.addExamplesPath('trainset/grayscale/val/gs_0', 0);
  await Classifier.addExamplesPath('trainset/grayscale/val/gs_1', 1);

  // Train
  await Classifier.train(config);

  // Save
  await Classifier.saveModel(
    `palmV${version}_D${config.denseUnits}_B${1 /
      config.batchSizeFraction}_Re-${Math.log(1 / config.learningRate) /
      Math.log(10)}_E${config.epochs}`
  );
})();
