require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('./load-csv')
const LinearRegression = require('./linear-regression.js')

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
    shuffle: true,
    splitTest: 0,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg'],
});

const regression = new LinearRegression(features, labels, {
    learningRate: 0.1,
    iterations: 400,
    batchSize: 10
  });
  regression.build();
  regression.train();
  // regression.predict([[120, 2, 380]]);

/*
  16.12,
  16.19,
  16.29,
  16.09,
  16.16
*/