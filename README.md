# Product_Review_Rating
This project aims to predict the helpfulness of any given product review.

## Python Libraries
Following libraries are required in this project:
* `json`
* `random`
* `nltk`
* `sklearn`
* `numpy`
* `pandas`
* `matplot`
* `time`

## Dataset
* Dataset can be found in [Amazon review data (2018)](http://deepyeti.ucsd.edu/jianmo/amazon/index.html), it is a json file.
* The model is trained based on Video Games category, but the code works on other categories too.

## Data Cleaning
* This project currently focus on the review content and vote number, so other unrelated data will be removed.
* The whole dataset is of 2,565,349 reviews which makes it time-consuming to analyse all the reviews. So part of the dataset will be randomly chosen to train the model.

## Model Training
* The main code for model training is in `Model_estimation.py`

## Trained Model Files
* In the main folder, all files end with `.pkl` are the trained models
* Models of cross-validation are too big, therefore these three are zipped in the file `cv-pkl-files.zip`

## Project Report
The final report is in `LING 131 Report.pdf`
