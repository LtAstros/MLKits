const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);
    this.labels = tf.tensor(labels);
    this.mseHistory = [];
    this.model = tf.sequential({
         layers: [
         tf.layers.dense({inputShape: [3], units: 1}),
         tf.layers.dense({units: 1})
         //tf.layers.dense({units: 10, activation: 'softmax'}),
         ]
       });
    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000 },
      options
    );

    this.weights = tf.zeros([this.features.shape[1], 1]);
  }

   build(){
      this.model.compile({
         optimizer: tf.train.adam(),
         loss: tf.losses.meanSquaredError,
         metrics: ['accuracy']
      })
   }

   debugger(){
    console.log(this.features.shape)
    this.features.print(true)
   }

  gradientDescent(features, labels) {
  }

   async train() {
   // await this.model.fit(this.features, this.labels, {
   //    epochs: 5,
   //    batchSize: 1,
   //    callbacks: (batch, logs) => {
   //       console.log('Accuracy', logs.acc);
   //    }
   // }).then(info => {
   //    console.log('Final accuracy', info.history.acc);
   // });
   //-8390.0194
   //-601.343
  }

  async predict(observations) {
   await this.model.fit(this.features, this.labels, {
      epochs: 500,
      batchSize: 32,
      callbacks: (batch, logs) => {
         console.log('Accuracy', logs.acc);
      }
   }).then(info => {
      console.log('Final accuracy', info.history.acc);
   });
   let obsTensor = this.processFeatures(observations)
   obsTensor.print()
   const prediction = this.model.predict(obsTensor);
   prediction.print();
  }

  test(testFeatures, testLabels) {
  }

  processFeatures(features) {
    features = tf.tensor(features);
    //features = tf.ones([features.shape[0], 1]).concat(features, 1);

    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5));
    } else {
      features = this.standardize(features);
    }

    return features;
  }

  standardize(features) {
    const { mean, variance } = tf.moments(features, 0);

    this.mean = mean;
    this.variance = variance;

    return features.sub(mean).div(variance.pow(0.5));
  }
}

module.exports = LinearRegression;