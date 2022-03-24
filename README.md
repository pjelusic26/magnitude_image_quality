# magnitude_image_quality

In image steganography, each data embedding process results in image quality degradation. For the same method, the amount of said degradation will be a direct result of signal implementation strength. Given that each image is different, it is impossible to use the same signal strength for each image.

The goal of this project is to develop a model that will be able to estimate the signal strength needed for the result image to be within a certain quality range. To do so, the dataset (consisted of 9545 512x512 grayscale images) will include:

- **Numerical data**
  - **magnitude coefficients** of the frequency domain
  - **PSNR values** after data embedding
  - signal strength **(gamma)**

- **Image data**
  - 32x32 grayscale images

As the output, the model will give an estimation of **gamma** needed for a fixed PSNR value.
