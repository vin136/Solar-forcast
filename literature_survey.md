# Papers Considered

[Benchmarking of deep learning irradiance forecasting models from sky images – An in-depth analysis](https://www.sciencedirect.com/science/article/pii/S0038092X21004266?casa_token=yOGIjVJ-DSwAAAAA:uUGNdXJfqqvG8XjrCb9yqUVm3cpn7YzCZSNetHOa6W1QwCxZCGSAFEipVr99hTMebhJHXVgyeIs)


Data:

Combines auxillary data: (GHI(t)(global horizontal irradiance), SZA(t)(solar zenith angle), cos(SZA(t)), sin(SZA(t)), SAA(t)(azimuthal angle), cos(SAA(t)), sin(SAA(t)).) with the SkyImager

Loss:

regularised MAE and MSE

Model:

- predict 10 min futer irradiance
- the CNN model was given five pairs of images, long and short exposures, taken every two minutes from time t to t − 8 min.

Studied various architectures 
1. CNN
2. CNN+LSTM
3. 3D-CNN
4. CONVLSTM

Conclusions:
1. Using LSTM(temporal info) inc performance(very slightly) but increasing the data gives best return.
2. forecast methods are always late relative to the ground truth. **models tend to behave like a very smart persistence model, avoiding large errors at the cost of missing peaks and having regular time delays.**



Other details:

- The training set was then generated from 35,000 samples randomly chosen from the 320 available days of 2017 (January to November), the validation set from 10,000 samples from the 320 available days of 2018 (Mid-February to December with 9 consecutive missing days in September) and the test set from 10,000 samples from the 363 available days of 2019 (January to November).
- Images were cropped and downscaled through bilinear filtering from  pixels to a resolution of 128*128.

**Aside**
Key Problem: Not being able to use the same metric(non-differentiable) for the loss.

For assessing the detection of ramps in wind and solar energy forecasting, specific algorithms were designed:  for shape, the ramp score based on a piecewise linear approximation of the derivatives of time series; for temporal error estimation, the Temporal Distortion Index (TDI). But they cant' be differentiated.

Commentary:

1.Can we penalize/use different loss function ?

sol: This [NIPS(2019) PAPER](https://arxiv.org/abs/1909.09020), [code](https://github.com/vincent-leguen/DILATE)

<img width="908" alt="Screen Shot 2022-05-30 at 10 38 55 AM" src="https://user-images.githubusercontent.com/21222766/171014922-db1049b5-0f24-4dee-98d9-fca205151b7b.png">

Esentially instead of making a point prediction at `t+n`, we also predict few intermediate values. Using the path values we measure/quantify the error with ground truth - a. shape mismatch b. time-mismatch(lags are penalized). Can be implemented, though a bit non-trivial.

