# Human Image Segmentation with PyTorch and U-Net
 
## Overview

This project implements a binary human image segmentation pipeline using PyTorch. The objective is to learn a pixel-wise mapping from an RGB input image to a foreground mask that separates the human subject from the background. The notebook follows the full segmentation workflow: dataset inspection, paired image-mask preprocessing, synchronized augmentation, model construction with a pretrained convolutional encoder, training, validation, checkpointing, and visual inference.

The implementation uses a U-Net segmentation model from `segmentation-models-pytorch` and is trained on paired human images and ground-truth masks from the `Human-Segmentation-Dataset-master` dataset. The final predictions are produced as binary masks using sigmoid activation followed by a fixed threshold.

## Objectives

This project focuses on the following practical objectives for building an image segmentation system in PyTorch:

- Understand how image segmentation datasets are organized as paired image and mask files.
- Build a custom PyTorch `Dataset` for image-mask segmentation samples.
- Apply segmentation-safe augmentations so that images and masks are transformed consistently.
- Load a pretrained convolutional segmentation architecture for transfer learning.
- Define training and validation loops for a binary segmentation task.
- Evaluate model behavior using loss curves and qualitative mask visualizations.

## Dataset and Problem Formulation

The notebook clones the dataset from `https://github.com/parth1620/Human-Segmentation-Dataset-master.git` and reads metadata from `train.csv`. The CSV contains two main columns:

- `images`: path to the RGB training image.
- `masks`: path to the corresponding ground-truth segmentation mask.

The task is formulated as binary semantic segmentation. For every input image, the model predicts a single-channel mask where each pixel is classified as either human foreground or background. The mask is read in grayscale, converted to a channel-aware format, resized to a fixed spatial resolution, normalized to `[0, 1]`, and rounded to enforce binary targets.

The dataset is split with `train_test_split` using an 80/20 validation split and `random_state=42`, producing:

| Split | Samples |
| --- | ---: |
| Training | 232 |
| Validation | 58 |

## Core Technologies

| Component | Role in the Project |
| --- | --- |
| PyTorch | Tensor operations, model definition, training, validation, checkpointing, and inference |
| `segmentation-models-pytorch` | Provides the pretrained U-Net architecture and binary Dice loss |
| Albumentations | Applies image-mask augmentations for segmentation training |
| OpenCV | Reads RGB images and grayscale masks from disk |
| NumPy | Handles image and mask array transformations |
| pandas | Loads and inspects the dataset CSV metadata |
| scikit-learn | Creates the train-validation split |
| Matplotlib | Visualizes images, masks, and predictions |
| tqdm | Displays training and validation progress bars |

## Project Workflow

1. **Environment setup**

   The notebook installs the required segmentation libraries in a Colab GPU runtime, including `segmentation-models-pytorch`, `albumentations`, and `opencv-contrib-python`.

2. **Dataset loading and inspection**

   The human segmentation dataset is cloned, and `train.csv` is loaded with pandas. A sample row is inspected by reading the RGB image with OpenCV, converting BGR to RGB, reading the associated mask in grayscale, and displaying the image-mask pair side by side.

3. **Configuration**

   The main experiment settings are defined centrally:

   | Parameter | Value |
   | --- | --- |
   | Device | `cuda` |
   | Epochs | `25` |
   | Learning rate | `0.003` |
   | Image size | `320 x 320` |
   | Batch size | `16` |
   | Encoder | `timm-efficientnet-b0` |
   | Encoder weights | `imagenet` |

4. **Segmentation augmentation**

   Albumentations is used to define separate augmentation pipelines for training and validation. The training pipeline resizes samples to `320 x 320` and applies horizontal and vertical flips with probability `0.5`. The validation pipeline only resizes the image-mask pair. Because Albumentations receives both `image` and `mask`, spatial transformations remain aligned across the input image and its segmentation target.

5. **Custom dataset construction**

   The `SegmentationDataset` class extends `torch.utils.data.Dataset` and implements the standard `__len__` and `__getitem__` methods. For each index, it reads the image and mask paths from the dataframe, loads the image as RGB, loads the mask as grayscale, resizes both arrays to the configured image size, applies optional augmentations, transposes arrays from HWC to CHW layout, casts them to `float32`, scales image intensities by `255.0`, and converts the mask into a binary tensor.

6. **Batch loading**

   The dataset objects are wrapped with PyTorch `DataLoader` instances. The training loader uses shuffling, while the validation loader keeps deterministic ordering. With a batch size of `16`, the notebook reports `15` training batches and `4` validation batches.

   The batch tensor contract is:

   | Tensor | Shape | Meaning |
   | --- | --- | --- |
   | Images | `[16, 3, 320, 320]` | Batch of normalized RGB images |
   | Masks | `[16, 1, 320, 320]` | Batch of binary segmentation masks |

7. **Model definition**

   The model is defined as a PyTorch `nn.Module` wrapper around `smp.Unet`. It uses a `timm-efficientnet-b0` encoder initialized with ImageNet weights, accepts 3-channel RGB inputs, and produces 1-channel binary segmentation logits. The model returns raw logits during inference and returns both logits and training loss when ground-truth masks are supplied.

8. **Loss function**

   The training objective combines two complementary losses:

   - `DiceLoss(mode='binary')`, which directly optimizes mask overlap.
   - `BCEWithLogitsLoss`, which applies binary cross-entropy to raw logits in a numerically stable way.

   The final loss is the sum of Dice loss and BCE-with-logits loss.

9. **Training and validation**

   The optimizer is Adam with learning rate `0.003`. Each training step moves images and masks to CUDA, clears gradients, computes logits and loss, backpropagates, updates model weights, and accumulates average epoch loss. Validation runs under `torch.no_grad()` with the model in evaluation mode. The best checkpoint is saved whenever validation loss improves.

10. **Inference and visualization**

    The notebook loads `best_mode.pt`, selects validation samples, computes raw logits, applies sigmoid activation, thresholds probabilities at `0.5`, and visualizes the input image, ground-truth mask, and predicted mask.

## Model Architecture

The segmentation network is a U-Net model created with `segmentation_models_pytorch.Unet`. U-Net is suitable for dense prediction tasks because it combines encoder features that capture semantic context with decoder features that recover spatial detail. In this project, the encoder is `timm-efficientnet-b0`, pretrained on ImageNet, which provides transfer-learned visual features before task-specific segmentation training.

Model configuration:

| Setting | Value |
| --- | --- |
| Architecture | U-Net |
| Encoder backbone | `timm-efficientnet-b0` |
| Encoder initialization | ImageNet pretrained weights |
| Input channels | `3` |
| Output classes | `1` |
| Output activation in model | `None` |
| Inference activation | `sigmoid` |
| Prediction threshold | `0.5` |

Using `activation=None` keeps the forward pass output as raw logits. This is appropriate because `BCEWithLogitsLoss` expects logits directly, and sigmoid is applied explicitly only during inference.

## Training Configuration

The notebook trains for `25` epochs on CUDA using Adam. The model checkpointing strategy is based on validation loss: whenever the current validation loss is lower than the best previous validation loss, the model state dictionary is saved as `best_mode.pt`.

| Configuration Item | Value |
| --- | --- |
| Optimizer | Adam |
| Learning rate | `0.003` |
| Epochs | `25` |
| Batch size | `16` |
| Training batches | `15` |
| Validation batches | `4` |
| Loss | Binary Dice loss + BCE with logits |
| Checkpoint file | `best_mode.pt` |

This setup trains the model to optimize both pixel-level binary classification and foreground-mask overlap, which is important for segmentation tasks where the quality of object boundaries and region coverage matters.

## Results

The training logs show rapid validation improvement after the first few epochs. The best validation checkpoint is saved at epoch `15`.

| Training Stage | Epoch | Train Loss | Validation Loss |
| --- | ---: | ---: | ---: |
| Initial epoch | 1 | 0.8134 | 2.4740 |
| Best validation checkpoint | 15 | 0.1434 | 0.1496 |
| Final epoch | 25 | 0.0942 | 0.1838 |

From epoch 1 to the best checkpoint, validation loss decreased from `2.4740` to `0.1496`, an improvement of approximately `93.95%`. The final epoch achieved the lowest training loss, but the best saved model is selected from epoch 15 because it has the lowest validation loss.

Qualitative inference is performed on validation indices `38`, `3`, `14`, `18`, `35`, and `49`. For each sample, the notebook loads `best_mode.pt`, forwards the image through the model, applies sigmoid activation to convert logits into probabilities, thresholds the probability map at `0.5`, and displays the predicted mask alongside the original image and ground-truth mask. These visual comparisons verify that the trained model learned to produce human foreground masks rather than only minimizing a numeric loss.

## Key Takeaways

- Segmentation datasets require synchronized preprocessing for images and masks; geometric transforms must be applied consistently to both.
- A custom PyTorch `Dataset` provides precise control over image loading, mask loading, tensor formatting, normalization, and binary target conversion.
- Transfer learning with a pretrained EfficientNet encoder allows the U-Net model to start from strong visual features instead of training all representations from scratch.
- Combining Dice loss with BCE-with-logits balances region overlap quality with stable binary pixel classification.
- Validation-based checkpointing is necessary because the final training epoch is not always the best generalizing model.
- Qualitative mask visualization is an essential complement to loss tracking for image segmentation projects.
