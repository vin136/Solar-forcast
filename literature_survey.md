# Papers Considered

1. [Benchmarking of deep learning irradiance forecasting models from sky images – An in-depth analysis](https://www.sciencedirect.com/science/article/pii/S0038092X21004266?casa_token=yOGIjVJ-DSwAAAAA:uUGNdXJfqqvG8XjrCb9yqUVm3cpn7YzCZSNetHOa6W1QwCxZCGSAFEipVr99hTMebhJHXVgyeIs)


Data:

Combines auxillary data( SIRTA laboratory,three years from 2017 to 2019.): (GHI(t)(global horizontal irradiance), SZA(t)(solar zenith angle), cos(SZA(t)), sin(SZA(t)), SAA(t)(azimuthal angle), cos(SAA(t)), sin(SAA(t)).) with the SkyImager. 

Loss:

regularised MAE and MSE

Model:

- predict 10 min futer irradiance
- the CNN model was given five pairs of images, long and short exposures, taken every two minutes from time t to t − 8 min.
- Architecture: CNN for image, followed by concatenating the auxillary features.

Studied various architectures 
1. CNN
2. CNN+LSTM
3. 3D-CNN
4. CONVLSTM

Conclusions:
1. Using LSTM(or complex architectures) inc performance(very slightly)  but increasing the data gives best return.
2. forecast methods are always late relative to the ground truth. **models tend to behave like a very smart persistence model, avoiding large errors at the cost of missing peaks and having regular time delays.**

<img width="659" alt="Screen Shot 2022-06-02 at 9 59 44 AM" src="https://user-images.githubusercontent.com/21222766/171646618-4f9b966a-0f4e-47d9-932e-c2b5fad7b6a8.png">

Useful Points:

1. To assess the model’s performance, they used forecast skill (FS) metric based on the smart persistence model (SPM).
2. Two useful metrics aside from MSE. 

Ramp Metric : Identify sudden changes. Now measure the average slope difference between the forecast and ground truth.
Temporal Distortion Index: quantify temporal misalignment between two time series

TDI: The TDI is defined as the area between the optimal(DTW) and the identity path normalised by the area below the latter, which corresponds to the percentage of temporal distortion relative to the maximal distortion.


Other details:

- The training set was then generated from 35,000 samples randomly chosen from the 320 available days of 2017 (January to November), the validation set from 10,000 samples from the 320 available days of 2018 (Mid-February to December with 9 consecutive missing days in September) and the test set from 10,000 samples from the 363 available days of 2019 (January to November).
- Images were cropped and downscaled through bilinear filtering from  pixels to a resolution of 128*128.


**Comments**
The tendency of reacting to past observations rather than actively anticipating future events is already recognized and people did some things.(reactive than anticipatory.) Some work:

1. [ECLIPSE : Envisioning Cloud Induced Perturbations in Solar Energy](https://arxiv.org/abs/2104.12419), got [code](https://github.com/tcapelle/eclipse_pytorch)

Idea: A spatio-temporal neural network architecture that models cloud motion from sky images to not only predict future irradiance levels but also segmented images, which provide richer information on the local irradiance map. They got less TDI. But they have lot of components.

<img width="846" alt="Screen Shot 2022-06-02 at 10 12 21 AM" src="https://user-images.githubusercontent.com/21222766/171649233-f4a74812-dc34-41e3-93c7-e760c4006bf4.png">

- recursively predict future states
that are then regressed to future irradiance values (Figure 2).
Predicting a sequence of future values instead of a single
value allows our model to learn a representation that can
detect rapid changes in solar flux, due to cloud occlusion
for instance.

**comment**

- Seems overly complex and moreover for segmentation they used non-ml algo(Hybrid Thresholding Algorithm) to generate ground truth labels. Typically,more data would compensate for the need of auxillary loss.

- `Pre-train and finetune`: Popular paradigm in modern deep learning. Maybe we can make a pre-trained net(say on NREL dataset) and present the effect of pre-training.

**Aside**
2. Key Problem: Not being able to use the same metric(non-differentiable) for the loss.(NO one did this to my knowledge)

For assessing the detection of ramps in wind and solar energy forecasting, specific algorithms were designed:  for shape, the ramp score based on a piecewise linear approximation of the derivatives of time series; for temporal error estimation, the Temporal Distortion Index (TDI). But they cant' be differentiated.

Commentary:

1.Can we penalize/use different loss function ?

sol: This [NIPS(2019) PAPER](https://arxiv.org/abs/1909.09020), [code](https://github.com/vincent-leguen/DILATE)

<img width="908" alt="Screen Shot 2022-05-30 at 10 38 55 AM" src="https://user-images.githubusercontent.com/21222766/171014922-db1049b5-0f24-4dee-98d9-fca205151b7b.png">

Esentially instead of making a point prediction at `t+n`, we also predict few intermediate values. Using the path values we measure/quantify the error with ground truth - a. shape mismatch b. time-mismatch(lags are penalized). Can be implemented, though a bit non-trivial.(backprop is hard,custom implementation)


2. [Convolutional neural networks for intra-hour solar forecasting based on sky image sequences](https://www.sciencedirect.com/science/article/pii/S0306261921016639)
(very recent, in 2022,successor to solarnet paper). [CODE](https://github.com/fengcong1992/SolarNet)

**NOTE**: Authors published 3 papers with increasing sophistication, this is the last one.

Dataset:
- the National Renewable Energy Laboratory
(NREL) solar radiation research laboratory (SRRL) dataset. The SRRL
dataset is one of the largest publicly available datasets with both
total sky images and meteorological measurements. (can get from OpenSolar package)
- There are 155,644 data points after filtering out nighttime data
points and aligning images with numerical data. To ensure the success-
ful training and convincing verification, six years (i.e., from 2012-01-01
to 2017-12-31).

- sky images every 10-min.
- Numerical parameters include tem-
perature, relative humidity, wind speed, GHI, DNI, diffuse horizontal
irradiance (DHI), and atmospheric pressure.  

Target parametr:
𝐂𝐒𝐈 = 𝐆𝐇𝐈/𝐂𝐒𝐆𝐇𝐈

Models:

1.SCNN (vggnet based cnn)
2. 3DCNN
And a lot of baseline models (random forests etc), ANN's.

They have built six independent models span 1-hour with a 10-min lead time, a 10-min
resolution, and a 10-min update rate. Each forecasting model takes a
**sky image sequence** as its input and predicts a single future GHI value
at each forecasting issue time.

Results:
They did extensive hyperparemeter tuning. Most important are

a. image look back/lengths : i.e., $2^0$, $2^1$,$2^2$,$2^3$. (10 min resolution) -> 2 images as best.
b. similarly for resolution : medium res is best (128*128)

Verdict: Not great for cloudy weather conditions.

<img width="633" alt="Screen Shot 2022-05-30 at 1 13 54 PM" src="https://user-images.githubusercontent.com/21222766/171037023-c1579502-98b2-47f0-8d2a-1d28da768c96.png">

One good way is to provide uncertainty on these forecasts, hopefully they will identify these weather conditions apriori.

3. [History and trends in solar irradiance and PV power forecasting: A
preliminary assessment and review using text mining](https://www.sciencedirect.com/science/article/pii/S0038092X17310022)

A most review paper(2018) analyzing 100's of papers in the space.Here we consider the relevant sections : skyimager,ML based approaches for solar forecasting. Not too much relevant work. Some useful points

- Pointed out how MAE/MSE isn't the best metric for practical use.
- A slightly accepted metric is FS(forecast skill):
FS is computed by dividing the error
indicator for a particular model (e.g., RMSE or MAE) with the corresponding error indicator of a reference model (usually being the
season-adjusted persistence model). This fraction is then
subtracted from 1, so that forecasts better than the reference model
yield a positive skill. 

**Note**:Did'nt cover much about `skyimager` based methods, infact called for one such study/review.

Above papers seems to be the STATE OF ART in terms of sophistication. Below I revise some prelimnary ones (using DL/ML+skyimages):

1. [KloudNet: Deep Learning for Sky Image Analysis and Irradiance Forecasting](https://link.springer.com/chapter/10.1007/978-3-030-12939-2_37)

continuous irradiance was converted into
binary values using a clear sky index threshold - classification problem. Only classification acc are compared at different sites with baseline methods.

2. [3D-CNN-based feature extraction of ground-based cloud images for direct normal irradiance prediction](https://www.sciencedirect.com/science/article/pii/S0038092X19301082)

Extracts features with 3D-CNN and combines(regression) with other features. forecast skill of 17.06% for 10-minute ahead. Trained a cloud-classification model.(The clear-sky index is defined as the ratio between the measured DNI and the estimated theoretical clear-sky DNI)


## Other Relevant/classic papers

1. [Introducing the Temporal Distortion Index to perform a bidimensional analysis of renewable energy forecast](https://www.sciencedirect.com/science/article/pii/S0360544215014619)
2. [Convolutional Neural Networks applied to sky images for
short-term solar irradiance forecasting](https://arxiv.org/pdf/2005.11246.pdf): Very prelimnary. Used Image+auxillary data.
3.[Deep Learning Based Multistep Solar Forecasting for PV Ramp-Rate Control Using Sky Images](https://ieeexplore.ieee.org/abstract/document/9072298)

Gives a multistep forecast. Takes in multiple images + auxillary data.

Data: The Solar Radiation Research Laboratory of the National
Renewable Energy Laboratory (NREL). 1 year data at 10min resoultion.
They just threw classic models (RESNET,VGG ETC)(used pretrained models)

Some quirks:
-  They stacked and compared for each channel. Xb has the worst forecasting accuracy. As a result, the red
channel of each image is used for the image stacking.
- The MSF model integrates a linear combination of five ResNet18 models that are individually trained on datasets of different
forecast horizons H from 5 to 9 min to provide 1-min forecast
resolution (R1min) for the control criteria. The forecasts are updated every 5 min (U 5min).

4. [A Deep Physical Model for Solar Irradiance Forecasting with Fisheye Images](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w38/Le_Guen_A_Deep_Physical_Model_for_Solar_Irradiance_Forecasting_With_Fisheye_CVPRW_2020_paper.pdf) (**CVPR paper**)

Deep learning + partial diff equations. (Didn't read). But has `high forecast skill score.`(esentially learns the coefficients of the PDE)


--------
Notes:

- Aside from the generic loss/metric(MSE/MAE) track the other domain specific params like TDI,RI.
- Once trained give slice based metrics.(how the model is behaving at various irradiance levels)-> this will give us ideas to improve the model.

Questions:

- It seems everyone is using a public dataset(reproducability), if we are using our dataset we either got to implement some of the paper's model and check on our dataset (to benchmark and say our's is better etc) or ?

