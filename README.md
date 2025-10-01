# 📓 MNIST Embeddings — Similarity & Retrieval

## Overview
This project builds a lightweight pipeline to generate, visualize, and explore **image embeddings** from the MNIST handwritten digit dataset. Using a pre-trained VGG16 backbone, we project digits into a 128-dimensional embedding space and analyze their structure via dimensionality reduction, similarity search, and a simple nearest-neighbor evaluation.

Goal: demonstrate how embeddings capture visual similarity and can support retrieval/classification tasks, using minimal and interpretable code.

## Data
- Dataset: [MNIST](http://yann.lecun.com/exdb/mnist/) (70k grayscale images of handwritten digits, 28x28 px).  
- Preprocessing: normalization to [0,1], channel expansion to 3 (RGB), resize to 32x32 for VGG16 compatibility.  
- Sample size: 5,000 digits used for training/embedding.

## Pipeline Summary
1. **Imports** — libraries as TensorFlow/Keras, NumPy, matplotlib, scikit-learn.  
2. **Data Loading & Preprocessing** — normalize, reshape, resize images.  
3. **Embedding Model** — VGG16 (imagenet weights, no top), dense layer (128-dim embedding).  
4. **Visualization** — dimensionality reduction with t-SNE, 2D scatterplot of embeddings.  
5. **Exploration** — quick look at structure across digit classes.  
6. **Similarity Index** — build a k-NN index with scikit-learn.  
7. **Similarity Search & Retrieval** — query image vs. nearest neighbors in embedding space.  
8. **Evaluation** — 1-NN accuracy on a held-out test split.

## Results & Insights
- Embeddings cluster by digit class in 2D t-SNE space.  
- Simple k-NN retrieval brings visually coherent neighbors.  
- 1-NN classifier achieves ~87% accuracy on embeddings (train=4000, test=1000).  
- Demonstrates embeddings are meaningful representations, even without task-specific fine-tuning.

## Conclusion
This notebook provides a concise demonstration of generating and exploring embeddings for MNIST digits. It illustrates the use of transfer learning, similarity search, and simple evaluation metrics in a compact pipeline.
