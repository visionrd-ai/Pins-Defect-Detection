# Pins-Defect-Detection

## Introduction
The use case is a car’s fuse box with over 60 fuses connected to different components, and usually 3 pin terminals all with different shapes and sizes. The fuse box in test/fusebox.jpg shows 3 of them, each of which has a different number and shape of pins.
![image](https://github.com/visionrd-ai/Pins-Defect-Detection/assets/87422803/8414860f-e932-4d3f-ae4e-5ce27b28718b)
![image](https://github.com/visionrd-ai/Pins-Defect-Detection/assets/87422803/41433d96-1557-4a70-b891-ceb7792e685f)
![image](https://github.com/visionrd-ai/Pins-Defect-Detection/assets/87422803/3bb7976f-19ea-45f9-a6a3-8ba2609e9460)

## Setup

Run the following command in the terminal.

```pip3 install -r requirements.txt```

For accelerated inference, TensorRT 8.6.1 is used. Our experimental setup is as follows.
  - Ubuntu 22.04.2 LTS x86_64
  - NVIDIA GeForce RTX 3060 Ti

## Training

Provide images and their corresponding masks in data/imgs and data/masks respectively. Run the following command to start training.

```python3 train.py --epochs 50```

## Inference

Run the following command to test your models.

```python3 predict_trt.py --input <your_test_image>```

## Create Your Own
A dataset of ~100 images, 30 for each terminal labelled with “Good” and “Not Good” masks is used. Although it is crucial to ensure that the lighting conditions and camera focus are pitch-perfect, the dataset overall should represent everything the model should expect to see in production. For this purpose, we use Albumentations to add random translation, rotation, and scaling to the training data. Once the dataset is ready, follow training instructions as before.

## More
Read the [blog]([url](https://visionrdai.com/home/blog/13)https://visionrdai.com/home/blog/13)!
