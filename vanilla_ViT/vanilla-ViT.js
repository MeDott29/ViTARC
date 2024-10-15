// Import necessary libraries
const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const { createCanvas, loadImage } = require('canvas');
const GIFEncoder = require('gifencoder');

// Vision Transformer Class
class VisionTransformer {
  constructor(imgSize = 224, patchSize = 16, numClasses = 1000, dim = 768, depth = 12, heads = 12, mlpDim = 3072, dropout = 0.1) {
    if (imgSize % patchSize !== 0) {
      throw new Error("Image dimensions must be divisible by the patch size");
    }
    this.imgSize = imgSize;
    this.patchSize = patchSize;
    this.numPatches = (imgSize / patchSize) ** 2;
    this.patchDim = 3 * patchSize ** 2; // 3 channels (RGB)
    this.dim = dim;
    this.numClasses = numClasses;

    // Learnable parameters
    this.patchEmbedding = tf.layers.dense({ units: dim });
    this.positionEmbeddings = tf.variable(tf.randomNormal([this.numPatches + 1, dim]));
    this.clsToken = tf.variable(tf.randomNormal([1, 1, dim]));
    this.transformerLayers = Array.from({ length: depth }).map(() =>
      tf.layers.multiHeadAttention({ numHeads: heads, keyDim: dim, dropoutRate: dropout })
    );
    this.mlpHead = tf.sequential({
      layers: [
        tf.layers.layerNormalization({ axis: -1 }),
        tf.layers.dense({ units: numClasses })
      ]
    });
    console.log("VisionTransformer initialized with imgSize=" + imgSize + ", patchSize=" + patchSize + ", numClasses=" + numClasses + ", dim=" + dim + ", depth=" + depth + ", heads=" + heads + ", mlpDim=" + mlpDim + ", dropout=" + dropout);
  }

  forward(input) {
    console.log("Input shape:", input.shape);
    const patches = this.extractPatches(input);
    console.log("Patches shape:", patches.shape);
    const x = this.patchEmbedding.apply(patches);
    console.log("Patch embeddings shape:", x.shape);
    const clsTokens = this.clsToken.tile([input.shape[0], 1, 1]);
    console.log("CLS token shape:", clsTokens.shape);
    let embeddings = tf.concat([clsTokens, x], 1);
    console.log("Concatenated CLS token and patches shape:", embeddings.shape);
    embeddings = embeddings.add(this.positionEmbeddings);
    console.log("Added positional embeddings shape:", embeddings.shape);
    let output = embeddings;
    this.transformerLayers.forEach((layer, index) => {
      output = layer.apply([output, output, output]);
      console.log(`Transformer layer ${index + 1} output shape:`, output.shape);
    });
    const clsOutput = output.slice([0, 0, 0], [-1, 1, -1]).reshape([-1, this.dim]);
    console.log("CLS output shape:", clsOutput.shape);
    return this.mlpHead.apply(clsOutput);
  }

  extractPatches(input) {
    return tf.tidy(() => {
      const patches = [];
      for (let y = 0; y < this.imgSize; y += this.patchSize) {
        for (let x = 0; x < this.imgSize; x += this.patchSize) {
          patches.push(input.slice([0, y, x, 0], [-1, this.patchSize, this.patchSize, 3]));
        }
      }
      return tf.concat(patches, 0).reshape([input.shape[0], this.numPatches, this.patchDim]);
    });
  }
}

// Example usage
const model = new VisionTransformer();
const input = tf.randomNormal([1, 224, 224, 3]);
const logits = model.forward(input);
logits.print();

// Create a 3-second animation of model outputs
const numFrames = 90; // 3 seconds at 30 frames per second
const encoder = new GIFEncoder(800, 400);
encoder.createReadStream().pipe(fs.createWriteStream('model_output_animation.gif'));
encoder.start();
encoder.setRepeat(0); // loop indefinitely
encoder.setDelay(100); // frame delay
encoder.setQuality(10); // image quality

(async () => {
  for (let i = 0; i < numFrames; i++) {
    console.log(`Generating frame ${i + 1}/${numFrames}...`);
    const input = tf.randomNormal([1, 224, 224, 3]);
    const logits = model.forward(input).dataSync();

    const canvas = createCanvas(800, 400);
    const ctx = canvas.getContext('2d');

    // Draw the logits as a bar chart
    ctx.fillStyle = 'blue';
    const barWidth = 800 / logits.length;
    for (let j = 0; j < logits.length; j++) {
      const barHeight = Math.min(400, Math.max(-400, logits[j] * 10));
      ctx.fillRect(j * barWidth, 400 - barHeight, barWidth - 1, barHeight);
    }

    encoder.addFrame(ctx);
  }

  encoder.finish();
  console.log("Animation saved as model_output_animation.gif");
})();