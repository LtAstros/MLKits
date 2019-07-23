const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegression {
   constructor(features, labels, options) {
      this.features = this.processFeatures(features);
      this.labels = tf.tensor(labels);
      this.mseHistory = [];
      this.model = tf.sequential({
         layers: [
         tf.layers.dense({inputShape: [this.features.shape[1]], units: 1}),
         tf.layers.dense({units: 1})
         //tf.layers.dense({units: 10, activation: 'softmax'}),
         ]
      });
      this.options = Object.assign(
         { learningRate: 0.1, iterations: 100, batchSize: 32 },
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

   async train() {
      await this.model.fit(this.features, this.labels, {
         epochs: this.options.iterations,
         batchSize: this.options.batchSize,
         callbacks: (batch, logs) => {
            console.log('Accuracy', logs.acc);
         }
      }).then(info => {
         console.log('Final accuracy', info.history.acc);
      });
      await this.model.save('file://./ml-models');
  }

   async predict(observations) {
      const loadedModel = await tf.loadModel('file://./ml-models/model.json')
      let obsTensor = this.processFeatures(observations)
      obsTensor.print()
      const prediction = loadedModel.predict(obsTensor);
      prediction.print();
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