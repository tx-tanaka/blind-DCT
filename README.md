# Blind DCT/DFT

Blind reconstruction of the original grayscale images from the magnitude-only discrete cosine transform (DCT) or discrete Fourier Transform (DFT) arrays.

Original image             |  Fourier Transform Intensity |  Reconstructed Image  
:-------------------------:|:-------------------------:|:-------------------------:
<img width="200" height="200" alt="image_10100" src="https://github.com/user-attachments/assets/14c7bd75-5dba-438e-8bf6-a6785c995af9" />  |  <img width="200" height="200" alt="output" src="https://github.com/user-attachments/assets/7aa32e1e-7663-45dd-a61a-a16ff4f98179" /> | <img width="200" height="200" alt="image_10100_dft_pred" src="https://github.com/user-attachments/assets/5b61c602-782c-46dd-8faf-cd7dc0034d74" />


1. Download image samples 

Run `collect_images.ipynb` to download grayscale image samples from [Lorem Picsum](https://picsum.photos). This operation downloads 10,000 samples of 200x200 grayscale image data for training, and 100 samples of 200x200 grayscale image data for testing. They will be saved in `img200x200/` and `img200x200_test/`. 

## Blind inverse DCT experiment

2. Generate magnitude-only DCT arrays

Run `img_to_absdct.ipynb` to generate magnitude-only DCT arrays corresponding to the downloaded training and testing data. They will be saved in  `abs_dct200x200/` and `abs_dct200x200_test/`.

3. Training

Run `train_absdct_to_image.ipynb` to train a U-Net that predicts the original image from the corresponding magnitude-only DCT array. The datasets in `img200x200/` and `abs_dct200x200/` will be used for training. After the training, the trained model is used to predict the original test images from their magnitude-only DCT arrays. In `img200x200_test/`, the original test images are named `image_*****.jpg` and the predicted images are saved as `image_*****_dct_pred.jpg`.

The trained network is saved as `unet_attention_absdct_to_image.pt`.

A CUDA-enabled virtual environment is recommended. The code will run on CPUs, but it will take many hours. The training time can also be controlled by adjusting the number of training epochs:
```
NUM_EPOCHS = 200
```

## Blind inverse DFT experiment

2. Generate magnitude-only DFT arrays

Run `img_to_absdft.ipynb` to generate magnitude-only DFT arrays corresponding to the downloaded training and testing data. They will be saved in  `abs_dft200x200/` and `abs_dft200x200_test/`.

3. Training

Run `train_absdft_to_image.ipynb` to train a U-Net that predicts the original image from the corresponding magnitude-only DCT array. The datasets in `img200x200/` and `abs_dft200x200/` will be used for training. After the training, the trained model is used to predict the original test images from their magnitude-only DFT arrays. In `img200x200_test/`, the original test images are named `image_*****.jpg` and the predicted images are saved as `image_*****_dft_pred.jpg`.

The trained network is saved as `unet_attention_absdft_to_image.pt`.

A CUDA-enabled virtual environment is recommended. The code will run on CPUs, but it will take many hours. The training time can also be controlled by adjusting the number of training epochs:
```
NUM_EPOCHS = 200
```
