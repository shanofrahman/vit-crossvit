# Vision Transformer (ViT) and CrossViT Implementation

This repository contains PyTorch implementations of Vision Transformer (ViT) and CrossViT architectures for image classification. The code is organized into the following components:

## ViT (Vision Transformer)

The ViT model consists of the following components:

- `ImageEmbedder`: Embeds image patches using a linear layer after layer normalization.
- `ViT`: The main ViT model, which includes patch embedding, positional encoding, transformer blocks, and a multi-layer perceptron (MLP) head for classification.

To use ViT for image classification, you can instantiate the `ViT` class, specifying parameters such as image size, patch size, number of classes, model dimension, depth, number of attention heads, MLP dimension, dropout rates, and more.

## CrossViT (Cross-Scale Vision Transformer)

The CrossViT model extends ViT by introducing cross-scale attention between small and large image patches. It includes additional components such as `MultiScaleEncoder`, `CrossTransformer`, and separate MLP heads for small and large patches.

To use CrossViT for image classification, you can instantiate the `CrossViT` class, specifying parameters for both small and large scales, including dimensions, patch sizes, encoder depths, attention heads, MLP dimensions, and more.

## Requirements

- PyTorch
- einops

## Usage

Clone the repository and use the provided classes in your PyTorch projects. Adjust the model parameters according to your specific task and dataset.

```bash
git clone https://github.com/yourusername/vision-transformer.git
cd vision-transformer
```

Refer to the example usage code provided in the `vit.py` file.

## License

This code is provided under the MIT License. See the `LICENSE` file for details.
