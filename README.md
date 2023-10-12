<p align="center">
  <img width="700" src="https://github.com/visionrd-ai/.github/assets/145563962/79a92550-c2e4-49f3-8229-bfe6545e54ea"></a>
</p>


<div align="center">


At [VisionRD](https://visionrdai.com/), we are utilizing cutting-edge artificial intelligence (AI) technologies to carry out accurate and effective quality inspections during the manufacturing process, resulting in a 50% reduction in time and a 90% improvement in the quality of car parts inspection.

VisionRD is also working on innovative products like ADAS/AD and IntelliSentry that will enhance the safety of driving in the future.

We welcome [contributions](https://github.com/visionrd-ai/Pins-Defect-Detection) from the community.
Join us on our mission of driving Innovation and Efficiency in the Automotive Industry.

[<img alt="alt_text" width="40px" src="utils/Linkedin.png" />](https://www.linkedin.com/company/visionrd-ai/)
[<img alt="alt_text" width="40px" src="utils/Instagram.png" />](https://www.instagram.com/visionrdai/)
[<img alt="alt_text" width="40px" src="utils/Facebook.png" />](https://www.facebook.com/visionrdai/)
[<img alt="alt_text" width="40px" src="utils/Twitter.png" />](https://twitter.com/Visionrd_ai/)
[<img alt="alt_text" width="40px" src="utils/YouTube.png" />](https://www.youtube.com/@Visionrdai/)

# Pins-Defect-Detection

## Introduction
The use case is a car’s fuse box with over 60 fuses connected to different components, and usually 3 pin terminals all with different shapes and sizes. The fuse box in test/fusebox.jpg shows 3 of them, each of which has a different number and shape of pins.


### Some example masks (All, Not Good, Good)

<p align="center">
  <img src="https://github.com/visionrd-ai/Pins-Defect-Detection/assets/87422803/8414860f-e932-4d3f-ae4e-5ce27b28718b" />
  <img src="https://github.com/visionrd-ai/Pins-Defect-Detection/assets/87422803/41433d96-1557-4a70-b891-ceb7792e685f" />
  <img src="https://github.com/visionrd-ai/Pins-Defect-Detection/assets/87422803/3bb7976f-19ea-45f9-a6a3-8ba2609e9460" />
</p>

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

### Results

<p align="center">
  <img src="https://github.com/visionrd-ai/Pins-Defect-Detection/assets/87422803/61282665-7290-4362-8a47-9361509bd31f" />
  <img src="https://github.com/visionrd-ai/Pins-Defect-Detection/assets/87422803/d85eb42e-eaaf-4eff-8c65-e9cf5e8c16ae" />
  <img src="https://github.com/visionrd-ai/Pins-Defect-Detection/assets/87422803/feef03a8-3eb8-4d74-ad3d-285f4973e7f3" />
</p>


## Create Your Own
A dataset of ~100 images, 30 for each terminal labelled with “Good” and “Not Good” masks is used. Although it is crucial to ensure that the lighting conditions and camera focus are pitch-perfect, the dataset overall should represent everything the model should expect to see in production. For this purpose, we use Albumentations to add random translation, rotation, and scaling to the training data. Once the dataset is ready, follow training instructions as before.

## More
Read the [blog](https://visionrdai.com/home/blog/13) or visit our [website](https://visionrdai.com/) for more!
