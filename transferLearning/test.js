const { Classifier } = require('../mobilenetClassifier');

(async () => {
  await Classifier.init();

  await Classifier.loadModel('palmSplitV2_D1024_B01_Re-4_E50');

  await Classifier.testPath('trainset/grayscale/val/gs_0', 0);
  await Classifier.testPath('trainset/grayscale/val/gs_1', 1);
})();
