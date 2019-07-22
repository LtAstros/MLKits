require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('./load-csv')
const LinearRegression = require('./linear-regression.js')

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg'],
});

const regression = new LinearRegression(features, labels, {
    learningRate: 0.1,
    iterations: 3,
    batchSize: 10
  });
  console.log("building")
  regression.build();
  console.log("training")
  regression.train();
  console.log("predicting")
  regression.predict([[120, 2, 380]]);
//   console.log("debugger")
//   regression.debugger();