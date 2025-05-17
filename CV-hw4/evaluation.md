# Semantic Segmentation Evaluation Methods and Resolution Impact Analysis

## For model evaluation, you can run the following command to test the model:
```bash
python test_deeplabv3plus.py --checkpoint <your deeplabv3plus model path> # in resolution 512x1024
# or test the model in original resolution:
python test_deeplabv3plus_origin_resolution.py --checkpoint <your deeplabv3plus model path>
# in resolution 1024x2048, will only loss about 0.06% mIoU, this still inference in 512x1024 resolution for better performance but upsample to 1024x2048 resolution for evaluation
```


## Evaluation Metrics

In semantic segmentation tasks, the primary evaluation metrics are **Intersection over Union (IoU)** and **mean Intersection over Union (mIoU)**.

### Intersection over Union (IoU)

IoU is a metric used to evaluate segmentation quality by calculating the ratio of the intersection to the union of the predicted region and the ground truth region:

$$IoU = \frac{TP}{TP + FP + FN}$$

Where:
- TP (True Positive): Number of pixels correctly predicted as the target class
- FP (False Positive): Number of pixels incorrectly predicted as the target class
- FN (False Negative): Number of pixels incorrectly predicted as other classes

IoU values range from 0 to 1, with values closer to 1 indicating better segmentation performance.

### Mean Intersection over Union (mIoU)

mIoU is the standard evaluation metric for multi-class segmentation tasks, calculated as the average of IoU values across all classes:

$$mIoU = \frac{1}{N}\sum_{i=1}^{N}IoU_i$$

Where N is the number of classes.

## Confusion Matrix and IoU Calculation

A confusion matrix is an N × N matrix (where N is the number of classes), with each element (i,j) representing the number of pixels that belong to class i but were predicted as class j.

Based on the confusion matrix, we can calculate IoU for each class:

```
For class i:
TP = confusion_matrix[i,i]
FP = sum(confusion_matrix[:,i]) - confusion_matrix[i,i]
FN = sum(confusion_matrix[i,:]) - confusion_matrix[i,i]
IoU_i = TP / (TP + FP + FN)
```

## Impact of Input Resolution on Semantic Segmentation Performance

In semantic segmentation models, the resolution of input images significantly affects model performance. Here are the main factors:

### 1. Precision Loss Due to Downsampling

Deep learning models typically use fixed input dimensions (e.g., 512×1024 pixels) for training and inference. For images with higher resolution, downsampling is necessary, which may lead to:

- **Loss of small objects**: Small target objects may become extremely small or completely disappear after downsampling
- **Blurred boundaries**: Object boundary details become unclear during resolution reduction
- **Loss of texture information**: Recognition of certain classes depends on texture details, which may be lost during downsampling


### 2. Relationship Between Resolution and mIoU

Generally, the relationship between input resolution and mIoU shows the following characteristics:

- **Higher resolution typically improves mIoU**: More details are preserved, especially for small objects and fine boundaries
- **Increased computational complexity**: Higher resolutions significantly increase memory usage and computational load
- **Optimal balance point**: There exists an optimal resolution beyond which mIoU improvements become minimal

## Original Resolution vs. Unified Resolution: Evaluation Strategy Comparison

### Strategy 1: Unified Resolution Evaluation (In the training and testing, I use it for less computational cost)

In this strategy, all images are resized to the same resolution (e.g., 512×1024), which is the resolution used during model training. Advantages include:

- **Computational efficiency**: Batch processing is more efficient
- **Consistency with training conditions**: Avoids distribution shift
- **Memory-friendly**: Prevents memory overflow issues for high-resolution images

Disadvantages include:

- **Does not reflect real-world performance**: In practical applications, we typically need to process images of various resolutions
- **Disadvantageous for small object classes**: Small classes may lose information during scaling

### Strategy 2: Original Resolution Evaluation (In the final Evaluation, I use it. And it only loss about *0.06%* mIoU)

In this strategy, the model processes images at a fixed resolution (e.g., 512×1024), but prediction results are upsampled to the original image resolution (here 1024×2048) for evaluation:

- **Advantages**: Better reflects model performance in real-world application scenarios
- **Disadvantages**: Requires additional post-processing steps, and upsampling may introduce artifacts


## Resolution Impact on the Cityscapes Dataset

### Understanding Dimension Conventions

Before discussing resolution impact, it's crucial to understand the different dimension ordering conventions used by various libraries:

- **PIL (Python Imaging Library)**: Uses the convention (width, height)
- **NumPy and PyTorch**: Use the convention (height, width, [channels])

This distinction is important when discussing image resolutions and can be a source of confusion when working with different libraries.

### Original and Input Resolutions

The Cityscapes dataset has the following characteristics:

- **Original images**: 2048×1024 pixels (width×height in PIL convention)
- **Model input**: 1024×512 pixels (width×height in PIL convention)

When expressed in NumPy/PyTorch convention (height×width), these become:
- **Original images**: 1024×2048 pixels
- **Model input**: 512×1024 pixels

Throughout our codebase, we primarily use the height×width (NumPy/PyTorch) convention when configuring model parameters, which is why you'll see "512×1024" in the configuration files. However, when using PIL operations like `image.size` or `image.resize()`, width comes first.

### Scaling Impact

When downsampling from original to input resolution:

1. **Horizontal scaling ratio**: 2:1 (width reduced from 2048 to 1024)
2. **Vertical scaling ratio**: 2:1 (height reduced from 1024 to 512)

This uniform scaling (2:1 in both dimensions) preserves the aspect ratio of the images, avoiding distortion while reducing resolution by a factor of 4 overall (2× in each dimension).


## Resolution Management in Semantic Segmentation Systems

Our semantic segmentation system follows a specific resolution management approach:

1. **Training**: The model is trained on images resized to a fixed resolution of 512×1024 (height×width)
2. **Inference**: During testing, input images are also resized to 512×1024 (height×width)
3. **Training Validation**: Still performed at 512×1024 resolution for speed and efficiency
4. **Final Evaluation**: Predictions are upsampled to the original resolution (typically 1024×2048 height×width) for IoU calculation

This approach balances computational efficiency with evaluation accuracy. The model learns and predicts at a manageable resolution, while evaluation happens at full resolution to maintain ground truth precision.

## Dimension Ordering Considerations in Image Processing

When processing image resolutions, it's important to note the differences in dimension ordering across various libraries and functions:

### Dimension Representation Conventions

Understanding dimension ordering is critical for debugging and avoiding errors in image processing pipelines:

1. **PIL (Python Imaging Library)**:
   - `image.size` returns `(width, height)` format
   - Example: `(2048, 1024)` means width 2048 pixels, height 1024 pixels
   - PIL's resize function takes parameters as `resize((width, height))`

2. **NumPy Arrays**:
   - `array.shape` returns `(height, width, channels)` format
   - Example: `(1024, 2048, 3)` means height 1024 pixels, width 2048 pixels, 3 color channels
   - This is the opposite order from PIL's convention!

3. **PyTorch Tensors**:
   - Typically use `(batch_size, channels, height, width)` format
   - Example: `(1, 3, 512, 1024)` means 1 image, 3 channels, height 512 pixels, width 1024 pixels
   - Follows the same height-width ordering as NumPy

### Dimension Changes During Processing Pipeline

The complete processing pipeline involves several dimension transformations:

1. **Original Image Loading (PIL)**: 
   - Size is `(2048, 1024)` (width × height)

2. **Resizing for Model Input (PIL)**: 
   - Size becomes `(1024, 512)` (width × height)
   - This maintains the 2:1 aspect ratio

3. **Conversion to Tensor (NumPy/PyTorch)**:
   - Shape becomes `(3, 512, 1024)` (channels, height, width)
   - Note the dimension order switch from width-height to height-width!

4. **Model Prediction (PyTorch)**: 
   - Output shape is `(batch_size, num_classes, 512, 1024)`

5. **Post-processing (NumPy)**:
   - Prediction shape is `(512, 1024)` (height, width)

6. **Upsampling to Original Resolution (PIL then NumPy)**:
   - PIL resize to `(2048, 1024)` (width, height)
   - Final NumPy array shape: `(1024, 2048)` (height, width)

### Practical Impact Example

In our semantic segmentation project's practical execution:

- Original Cityscapes images: `(2048, 1024)` (width × height in PIL)
- Model input size: `(1024, 512)` (width × height in PIL)
- Configuration settings: `[512, 1024]` (height × width in config, following NumPy convention)
- Low-resolution prediction: `(512, 1024)` (height × width, NumPy array)
- Upsampled prediction: `(1024, 2048)` (height × width, NumPy array)

This dimension ordering difference is not an error but a convention difference between libraries - predictions are made at 512×1024 (height×width) and then upsampled to the original resolution of 1024×2048 (height×width) for IoU calculation.

These differences don't represent any errors or distortions; they simply reflect different conventions across libraries. During debugging and visualization, always explicitly indicate which dimension ordering is being used to avoid confusion.


## Resolution Transformation Process in Implementation

Our semantic segmentation pipeline employs a specific resolution transformation workflow, which balances computational efficiency with evaluation accuracy. This section details the exact transformation processes implemented in our code.

### Resolution Change Workflow

The complete workflow of resolution changes in our testing pipeline is as follows:

1. **Original Image Loading**: 
   - Cityscapes images are loaded at their original resolution (2048×1024 pixels in width×height format)

2. **Downsampling for Model Input**:
   - Images are resized to 512×1024 (height×width) or 1024×512 (width×height) using **bilinear interpolation**
   - This is implemented via `transforms.Resize((512, 1024), interpolation=PIL.Image.BILINEAR)`
   - The downsampling ratio is 2:1 in both dimensions (original:input), preserving aspect ratio

3. **Model Prediction**:
   - The DeepLabV3+ model processes these downsampled images
   - Output resolution matches the input resolution: 512×1024 (height×width)

4. **Upsampling Predictions**:
   - Predictions are upsampled back to the original resolution of 1024×2048 (height×width)
   - This upsampling uses **nearest neighbor interpolation** to preserve the discrete class labels
   - Implementation: `pred_pil.resize(original_size, PIL.Image.NEAREST)`

5. **Evaluation Against Ground Truth**:
   - IoU is calculated by comparing the upsampled predictions against original-resolution ground truth
   - Ground truth is always maintained at original resolution with no resampling

### Interpolation Methods Rationale

Two different interpolation methods are used at different stages of the pipeline:

#### Bilinear Interpolation for Input Downsampling
```python
transforms.Resize((512, 1024), interpolation=PIL.Image.BILINEAR)
```

- **Why Bilinear**: Bilinear interpolation produces smoother results when downsampling RGB images
- **Advantages for Input**: 
  - Preserves gradients and edge information better than nearest neighbor
  - Reduces aliasing artifacts that could affect model performance
  - Appropriate for continuous data like RGB pixel values

#### Nearest Neighbor Interpolation for Output Upsampling
```python
pred_pil = pred_pil.resize(original_size, PIL.Image.NEAREST)
```

- **Why Nearest Neighbor**: Preserves the exact class assignments without introducing new values
- **Advantages for Segmentation Maps**:
  - Maintains discrete class labels without blending between classes
  - No interpolation artifacts like new invalid class IDs
  - Conceptually appropriate for categorical data

### Impact of Interpolation Methods on Segmentation Quality

The choice of interpolation method significantly affects the quality of semantic segmentation results, especially when there are resolution changes in the pipeline. Our implementation uses different interpolation methods at different stages, each with specific impacts on quality.

#### Bilinear Interpolation Characteristics
- **Method**: Calculates the weighted average of the 4 nearest pixel values based on distance
- **Mathematical representation**: Applies linear interpolation in both x and y directions
- **Impact on input images**:
  - Produces smoother transitions between pixels
  - Better preserves gradients and edge details during downsampling
  - May slightly blur fine details
  - Handles continuous data (RGB values) appropriately

#### Nearest Neighbor Interpolation Characteristics
- **Method**: Simply assigns each new pixel the value of the nearest original pixel
- **Mathematical representation**: Non-interpolating, discrete assignment
- **Impact on segmentation maps**:
  - Preserves exact class labels without creating invalid intermediate values
  - Results in jagged edges and "blocky" appearance after upsampling
  - May cause area distortion for very small objects
  - Cannot recover details lost during the initial downsampling

### Visual Artifacts from Interpolation

Different artifacts appear in our pipeline depending on the interpolation method:

1. **Input Downsampling with Bilinear** (RGB Image → Model Input)
   - Small objects or thin structures may become blurred or disappear
   - Fine boundary details become less distinct
   - Texture information is averaged out, potentially making certain classes harder to distinguish

2. **Output Upsampling with Nearest Neighbor** (Prediction → Original Size)
   - Produces stair-like patterns along diagonal boundaries
   - Creates blocky artifacts, especially for small objects
   - May cause area distortion for very small objects
   - Cannot recover details lost during the initial downsampling




### Alternative Interpolation Methods

While our implementation uses bilinear for input and nearest neighbor for output, other methods could be considered:

1. **Bicubic Interpolation**:
   - Could potentially preserve more details during input downsampling
   - Higher computational cost
   - Benefit may be marginal for our 2:1 downsampling ratio

2. **Area-Based Resampling**:
   - Would be more appropriate for downsampling input images
   - Takes the weighted average of all pixels that contribute to an output pixel
   - More accurate representation of the original image at lower resolution

3. **Advanced Upsampling Methods**:
   - Super-resolution techniques could potentially recover some details in upsampling
   - Would require significant additional computation
   - Not suitable for categorical data like segmentation maps

## Resolution Handling in Semantic Segmentation Frameworks

Different semantic segmentation frameworks may have their own conventions and best practices for handling image resolutions. Here, we discuss some specific considerations for popular frameworks.

### Framework-Specific Considerations

1. **TensorFlow/Keras**:
   - Use `tf.image.resize()` for resizing images, with `method` parameter to specify the interpolation method (e.g., `tf.image.ResizeMethod.BILINEAR`).
   - Be mindful of the data format (channels last vs. channels first) when configuring input pipelines.

2. **PyTorch**:
   - Use `torchvision.transforms.Resize()` for resizing, with `interpolation` parameter to specify the method (e.g., `transforms.InterpolationMode.BILINEAR`).
   - Ensure consistent use of dimension ordering (height, width) throughout the data processing and model configuration.

3. **OpenCV**:
   - Use `cv2.resize()` for resizing images, with `interpolation` parameter to specify the method (e.g., `cv2.INTER_LINEAR` for bilinear interpolation).
   - Be aware of the BGR color ordering used by OpenCV when converting images to/from other formats.

4. **Albumentations**:
   - Use `A.Resize()` for resizing images, with `interpolation` parameter to specify the method (e.g., `cv2.INTER_LINEAR`).
   - Albumentations also provides advanced augmentation techniques that can be used to further improve model robustness to resolution changes.

## Code Examples and Resolution Handling Best Practices

### Resolution Transformation Code Examples

Here are key code snippets from our implementation that handle resolution changes:

#### 1. Input Image Downsampling (Bilinear)

```python
def get_visualization_transform():
    """Get transform for visualization with fixed 512x1024 input (model's training resolution)"""
    return transforms.Compose([
        transforms.Resize((512, 1024), interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
```

#### 2. Prediction and Upsampling Process

```python
def predict_image(model, image_path, transform, device):
    # Load image at original resolution
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height) format in PIL
    
    # Apply transform for model input - WITH resizing to 512x1024
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction at low resolution
    with torch.no_grad():
        output = model(input_tensor)
        pred_lowres = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    # Upsample prediction to original resolution using nearest neighbor
    pred_pil = PIL.Image.fromarray(pred_lowres.astype(np.uint8))
    pred_pil = pred_pil.resize(original_size, PIL.Image.NEAREST)
    pred = np.array(pred_pil)
    
    return image, pred, pred_lowres
```

#### 3. Ground Truth Handling (No Resizing)

```python
def get_ground_truth(image_path, city_dir):
    # Extract path components
    parts = Path(image_path).parts
    city = parts[-2]
    file_name = parts[-1].replace('_leftImg8bit.png', '_gtFine_labelIds.png')
    
    # Construct label path
    label_path = os.path.join(city_dir, city, file_name)
    
    # Load label at original resolution (no resizing)
    label = Image.open(label_path)
    label_np = np.array(label, dtype=np.int64)
    
    # Map IDs to train IDs
    from datasets.cityscapes import id_to_trainid
    for k, v in id_to_trainid.items():
        label_np[label_np == k] = v
    
    # Handle ignore regions
    label_np[label_np == 255] = 255
    label_np[label_np == -1] = 255
    
    return label_np
```

## Ignore Label Handling in Semantic Segmentation Evaluation

Proper handling of ignore labels is crucial for accurate evaluation of semantic segmentation models. This section details how ignore labels are processed in our mIoU calculation and why this approach is standard and reasonable.

### Ignore Labels in Cityscapes Dataset

The Cityscapes dataset designates certain pixels as "ignore" regions that should not participate in model evaluation:

1. **Source of Ignore Labels**:
   - Some classes in Cityscapes are marked with `ignoreInEval=True` in the dataset definition
   - These classes are assigned special trainId values (255 or -1) to indicate they should be ignored
   - Examples include "void" areas, unlabeled regions, and certain ambiguous boundaries

2. **Unified Ignore Index**:
   - All ignore regions are standardized to use a single value (255) during preprocessing
   - Original label IDs are mapped to training IDs, with 255 and -1 both treated as ignore regions:
   ```python
   # Map IDs to train IDs
   from datasets.cityscapes import id_to_trainid
   for k, v in id_to_trainid.items():
       label_np[label_np == k] = v
   
   # Handle ignore regions
   label_np[label_np == 255] = 255  # Keep ignore regions as 255
   label_np[label_np == -1] = 255   # Map -1 to 255 for ignore
   ```

### mIoU Calculation with Ignore Regions

When calculating the mIoU metric, our implementation explicitly excludes pixels labeled as ignore regions:

1. **Confusion Matrix Calculation**:
   - Create a mask that filters out pixels with the ignore index (255)
   - Only valid (non-ignored) pixels are used to build the confusion matrix:
   ```python
   def calculate_confusion_matrix(pred, gt, num_classes, ignore_index=255):
       # Create mask for valid data, excluding ignore_index
       mask = (gt != ignore_index)
       
       # Extract valid targets and predictions
       valid_targets = gt[mask].flatten()
       valid_preds = pred[mask].flatten()
       
       # Ensure indices are in valid range
       valid_indices = (valid_targets < num_classes) & (valid_preds < num_classes)
       valid_targets = valid_targets[valid_indices]
       valid_preds = valid_preds[valid_indices]
       
       # Update confusion matrix with valid pixels only
       if len(valid_targets) > 0:
           np.add.at(confusion_matrix, (valid_targets, valid_preds), 1)
   ```

2. **IoU Calculation per Class**:
   - For each class, IoU is calculated from the confusion matrix using the standard formula:
   ```python
   def calculate_miou(confusion_matrix):
       iou_per_class = []
       for i in range(confusion_matrix.shape[0]):
           # True positives: diagonal elements
           tp = confusion_matrix[i, i]
           # False positives: sum of column i - true positives
           fp = confusion_matrix[:, i].sum() - tp
           # False negatives: sum of row i - true positives
           fn = confusion_matrix[i, :].sum() - tp
           
           # Calculate IoU if denominator is not zero
           if tp + fp + fn > 0:
               iou = tp / (tp + fp + fn)
           else:
               iou = 0.0
           iou_per_class.append(iou)
   ```

3. **Mean IoU Calculation**:
   - The final mIoU is the average of all per-class IoUs, ignoring classes with zero IoU:
   ```python
   # Calculate mean IoU (simple average)
   miou = np.mean([iou for iou in iou_per_class if iou > 0])
   ```

### Why Ignoring Certain Regions is Standard and Reasonable

The approach of excluding ignore regions from mIoU calculation is both standard in the field and reasonable for several key reasons:

1. **Prevents Evaluation Bias**:
   - Unlabeled or ambiguous areas should not affect model evaluation
   - Including these areas would unfairly penalize models for regions where correct classification is impossible or undefined

2. **Follows Dataset Guidelines**:
   - The Cityscapes dataset explicitly defines which regions should be ignored during evaluation
   - This approach respects the dataset creators' intentions and standardized benchmarking protocols

3. **Handles Class Boundary Uncertainty**:
   - Object boundaries are often subject to annotation uncertainty and ambiguity
   - Ignoring these regions prevents evaluation metrics from being unduly influenced by subjective boundary placement

4. **Focuses on Meaningful Classes**:
   - By ignoring irrelevant pixels, the evaluation concentrates on the classes that matter for the application
   - This provides a more accurate assessment of model performance on the task's actual objectives

5. **Standard Practice in Semantic Segmentation**:
   - This approach is widely adopted across semantic segmentation research
   - Major benchmarks like Cityscapes, PASCAL VOC, and ADE20K all employ similar ignore region handling

### Addressing Class Imbalance in Evaluation

Beyond ignore region handling, our implementation also addresses class imbalance through:

1. **Simple Average mIoU**:
   - Equal weight given to each class regardless of its frequency
   - Prevents dominant classes from overwhelming the metric

2. **Weighted Average mIoU**:
   - Alternative metric that weights classes by their pixel frequency
   - Provides a complementary perspective that may better reflect perceived visual quality
   ```python
   # Calculate weighted mIoU
   weight_sum = np.sum(valid_weights) + epsilon
   weighted_miou = np.sum(valid_ious * valid_weights) / weight_sum
   ```

This dual approach to mIoU calculation, combined with proper handling of ignore regions, ensures a comprehensive and fair evaluation of semantic segmentation performance across diverse urban scenes.
