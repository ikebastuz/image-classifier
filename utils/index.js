const fs = require('fs');
const util = require('util');
const { resolve } = require('path');
const { createCanvas, Image } = require('canvas');
const uuidv1 = require('uuid/v1');
const tf = require('@tensorflow/tfjs');

const stat = util.promisify(fs.stat);
const readDir = util.promisify(fs.readdir);
const saveImage = util.promisify(fs.writeFile);

async function addImageToDataset(imageData) {
  const base64Data = imageData.data.replace(/^data:image\/jpeg;base64,/, '');

  await saveImage(
    `trainset/${imageData.class}_${uuidv1()}.jpg`,
    base64Data,
    'base64'
  );
  return true;
}

async function getFileNames(path) {
  const subdirs = await readDir(path);
  const imageNames = await Promise.all(
    subdirs.map(async (subdir) => {
      const res = resolve(path, subdir);
      return (await stat(res)).isDirectory() ? getFiles(res) : res;
    })
  );

  return imageNames.filter((imageName) =>
    imageName.match(/[a-zA-Z0-9\s_\\.\-\(\):]+(.jpg|.jpeg|.png)/i)
  );
}

function getModelAccuracy(stats) {
  return Object.values(stats).reduce(
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
}

async function processImage(imageData, imageSize) {
  const canvas = createCanvas(imageSize.width, imageSize.height);
  const ctx = canvas.getContext('2d');
  const img = new Image();

  const promise = new Promise((resolve) => {
    img.crossOrigin = '';
    img.onload = () => {
      const { x, y } = getOffsets(img, imageSize);
      ctx.drawImage(img, x, y);
      resolve(canvas);
    };
  });

  img.src = `${imageData}`;
  return promise;
}

function getOffsets(img, imageSize) {
  let x, y;
  if (Number(img.width) > imageSize.width) {
    x = -(img.width - imageSize.width) / 2;
  } else {
    x = (imageSize.width - img.width) / 2;
  }

  if (Number(img.height) > imageSize.height) {
    y = -(img.height - imageSize.height) / 2;
  } else {
    y = (imageSize.height - img.height) / 2;
  }

  return { x, y };
}

function normalizeImage(imageData) {
  return tf.tidy(() => {
    const webcamImage = tf.fromPixels(imageData);
    const croppedImage = cropImage(webcamImage);
    const batchedImage = croppedImage.expandDims(0);

    return batchedImage
      .toFloat()
      .div(tf.scalar(127))
      .sub(tf.scalar(1));
  });
}

function cropImage(img) {
  const size = Math.min(img.shape[0], img.shape[1]);
  const centerHeight = img.shape[0] / 2;
  const beginHeight = centerHeight - size / 2;
  const centerWidth = img.shape[1] / 2;
  const beginWidth = centerWidth - size / 2;
  return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
}

module.exports = {
  getFileNames,
  getModelAccuracy,
  processImage,
  addImageToDataset,
  normalizeImage
};
