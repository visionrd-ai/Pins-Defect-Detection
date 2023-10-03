import matplotlib.pyplot as plt


def plot_img_and_mask(img, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()

import glob
import cv2
import numpy as np

def resize_all():

    images = glob.glob('data/imgs/*.png')
    for img in images:
        image_read = cv2.imread(img)
        image_read = cv2.resize(image_read, (139,352))
        cv2.imwrite(img, image_read)

    masks = glob.glob('data/masks/*.png')
    for mask in masks:
        mask_read = cv2.imread(mask)
        mask_read = cv2.resize(mask_read, (139,352))
        cv2.imwrite(mask, mask_read)

    print("Data resized to (180,354).")

# resize_all()
