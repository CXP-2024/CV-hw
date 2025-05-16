# Models

I first tried to use the UNet model, but it did not perform well for this segmentation task. I then implemented the DeepLabV3 model and further improved it to DeepLabV3+. The DeepLabV3+ architecture consists of the following key components:

* A backbone network using ResNet with atrous convolutions
* Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale context
* A decoder module that refines the segmentation results
* Skip connections that combine low-level features with high-level features

This architecture is particularly effective for semantic segmentation tasks as it captures both detailed spatial information and broad contextual information.

## Model Architecture Details

The DeepLabV3+ model I implemented consists of several carefully designed components:

### ResNet Backbone

I implemented a custom ResNet backbone with atrous (dilated) convolutions:

* Initial layer: 7×7 convolution with stride 2, followed by batch normalization, ReLU, and max pooling
* Layer 1: 3 ResNet blocks with 64 channels
* Layer 2: 4 ResNet blocks with 128 channels, stride 2
* Layer 3: 6 ResNet blocks with 256 channels, dilation rate 2
* Layer 4: 3 ResNet blocks with 512 channels, dilation rate 4

Each ResNet block consists of two 3×3 convolutional layers with batch normalization and ReLU, along with a residual connection. The increasing dilation rates in deeper layers ensure a larger receptive field without decreasing spatial resolution.

### ASPP Module

The Atrous Spatial Pyramid Pooling module consists of:

* One 1×1 convolution
* Three 3×3 atrous convolutions with dilation rates of 12, 24, and 36
* A global average pooling branch followed by a 1×1 convolution

These five branches are concatenated and fed through a 1×1 convolution with 256 output channels, followed by batch normalization, ReLU, and a dropout layer (rate=0.5) to obtain the final ASPP features.

### Decoder Module

The decoder integrates the semantically rich features from the ASPP module with spatially detailed low-level features from earlier layers:

* Low-level features from the first ResNet layer are processed by a 1×1 convolution to reduce channels to 48
* ASPP features are upsampled to match the spatial dimensions of the low-level features
* Both feature maps are concatenated and processed by two 3×3 convolutions
* The result is then upsampled to input resolution and fed to a final classifier

### Final Classification Layer

A simple 1×1 convolution transforms the decoder output into class logits with 19 channels (one for each class in the Cityscapes dataset).

### Weight Initialization

All convolutional layers use Kaiming initialization to ensure proper gradient flow during training, while batch normalization layers are initialized with weight=1 and bias=0.

## Model Input and Evaluation Resolution

To balance performance and computational efficiency, all training and evaluation were conducted at a consistent resolution of 512×1024 pixels. The original Cityscapes images (1024×2048) were resized to this working resolution during both training and testing phases.

This resolution choice was explicitly defined in the `config.yaml` file:

```
data:
  image_size: [512, 1024]  # height, width
```

Both input images and ground truth labels were resized to this resolution. During inference, the model predictions remain at 512×1024 resolution rather than being upscaled back to the original 1024×2048 resolution. This approach ensures consistency between training and evaluation conditions, while significantly reducing memory requirements and computational load.

The model architecture was designed to handle this specific resolution, with the encoder progressively reducing spatial dimensions and the decoder carefully upsampling features back to the input dimensions. The final interpolation layer in the DeepLabV3+ model upsamples the features to match the input resolution of 512×1024:

```
# Upscale to input resolution
x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
```

This consistent handling of resolution across both training and evaluation phases ensures fair comparisons and reliable performance metrics.

## Model Implementation Details

### Forward Pass Process

The forward pass through the model is implemented as follows:

1. Input image is processed by the ResNet backbone
2. Encoder produces high-level features (output of layer 4) and low-level features (output of layer 1)
3. High-level features are passed through the ASPP module for multi-scale context extraction
4. ASPP features and low-level features are combined in the decoder module
5. Final classifier converts the feature map to logits
6. Output is upsampled to input resolution using bilinear interpolation

### Key Implementation Components

#### Atrous (Dilated) Convolutions

I used dilated convolutions to increase the receptive field without increasing the number of parameters or reducing the spatial resolution. In standard convolutions, the filter elements are applied to adjacent input elements. In dilated convolutions, gaps ("holes") are introduced between filter elements, effectively increasing the receptive field while maintaining the same number of parameters.

For example, with a dilation rate of 2, each filter element applies to inputs that are 2 pixels apart, effectively doubling the receptive field without increasing the filter size. This is crucial for semantic segmentation where both global context and fine details need to be preserved.

#### ASPP Design Considerations

The ASPP module captures multi-scale information through parallel atrous convolutions with different dilation rates (12, 24, and 36). These different rates allow the network to capture context at various scales:

* Smaller dilation rates: capture fine details and local context
* Larger dilation rates: capture broader context and object relationships
* Global average pooling branch: captures image-level context

#### Skip Connection Implementation

The skip connection between the encoder and decoder is crucial for recovering spatial details lost during downsampling. By connecting the low-level features from the first ResNet block to the decoder, the model can combine semantic information (from deep layers) with spatial information (from shallow layers), resulting in more accurate boundary delineation.

The low-level features undergo a 1×1 convolution to reduce the channel dimension from 64 to 48, making the fusion more balanced and computationally efficient.

## Acknowledgements
This implementation is based on the original DeepLabV3+ paper and the Cityscapes dataset. The architecture and design choices are inspired by the work of Chen et al. (2018) in "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation" and the Cityscapes dataset documentation.