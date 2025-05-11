Since you have learned semantic segmentation task in computer vision, you are
required implement your own semantic segmentation models for this assignment.

This repository contains scripts for the inspection, preparation, and evaluation of the Cityscapes dataset. This large-scale dataset contains a diverse set of stereo video sequences recorded in street scenes from 50 different cities, with high quality pixel-level annotations of 5 000 frames in addition to a larger set of 20 000 weakly annotated frames.

Here are the requirements:
• Settings: Cross-validation is required for this project.
• Measurement: Mean Intersection over Union(MIoU) score is the only
measurement of the performance.
• Models: Feel free to choose semantic segmentation models (vanilla or
SOTA is ok). But we DO NOT recommend you to choose some large
networks, such as mask r-cnn.
• Experiments: If you use some techniques to improve the performance,
please do some ablation study for this changes compared with your
baseline model.
• Visualizations: Visualization of your results is required, and please put
down your experimental settings (hyperparameters) and training/testing
curves.
• Others: Design an original architecture is highly encouraged. (Bonus:
if it can work well)