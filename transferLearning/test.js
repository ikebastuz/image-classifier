const { Classifier } = require('../mobilenetClassifier');

(async () => {
  await Classifier.init();

  await Classifier.loadModel('palm1024softmax000001');

  await Classifier.testPath('testset/0', 0);
  await Classifier.testPath('testset/1', 1);
})();
