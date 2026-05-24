# Blind DCT

Blind reconstruction of the original grayscale images from the magnitude-only discrete cosine transform (DCT) arrays.

## Execution

1. Download image samples

Run `collectimages.ipynb` to download grayscale image samples from [Lorem Picsum](https://picsum.photos). This operation downloads 10,000 samples of 50x50 greyscale image data for training, and 100 samples of 50x50 greyscale image data for testing. They will be saved in `img50x50/` and `img50x50_test/`. 

2. Generate magnitude-only DCT arrays

Run `img2dct.ipynb` to generate magnitude-only DCT arrays corresponding to the downloaded training and testing data. They will be saved in  `abs_dct50x50/` and `abs_dct50x50_test/`.

3. Training

Run `train_absdct_to_image.ipynb` to train a U-Net that predicts the original image from the corresponding magnitude-only DCT array. The datasets in `img50x50/` and `abs_dct50x50/` will be used for training. After the training, the trained model is used to predict the original test images from their magnitude-only DCT arrays. In `img50x50_test/`, the original test images are named `image_*****.jpg` and the predicted images are saved as `image_*****_pred.jpg`.

A CUDA-enabled virtual environment is recommended. The code will run on CPUs, but it will take many hours. The training time can also be controlled by adjusting the number of training epochs:
```
NUM_EPOCHS = 100
```

The trained network is saved as `unet_attention_absdct_to_image.pt`.
