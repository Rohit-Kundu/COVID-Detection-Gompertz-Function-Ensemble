# COVID-Detection-Gompertz-Function-Ensemble
This is the official implementation of the paper titled "Fuzzy Rank-based Fusion of CNN Models using Gompertz Function for Screening COVID-19 CT-Scans" under peer review in "Nature- Scientific Reports".

Abstract: COVID-19 has crippled the world’s health-care systems, setting back the economy and taking the lives of several people. Although potential vaccines are being tested around the world, it will take a long time to reach every human being. Thus, there is a dire need for early and accurate detection of COVID-19 to prevent the spread of the disease, even more, with new mutations of the virus emerging and spreading globally. The current gold-standard RT-PCR test is only 71% sensitive and is a laborious test to perform, leading to the incapability of conducting the population-wide screening. To this end, in this paper, we propose an automated COVID-19 detection system that uses CT-scan images of the lungs for classification into COVID and Non-COVID cases. The proposed method applies an ensemble strategy that generates fuzzy ranks of the base classification models using the Gompertz function and fuses the decision scores of the base models adaptively to make the final predictions on the test cases. Three Transfer Learning-based CNN models have been used, namely VGG-11, Wide ResNet-50-2, and Inception v3, for generating the decision scores for fusion using the proposed ensemble model. The framework has been evaluated on two publicly available datasets achieving state-of-the-art performance, justifying the reliability of the model.

## Requirements

To install the dependencies, run the following using the command prompt:

`pip install -r requirements.txt`

## Running the code on the COVID data
In this repository we take the example of the SARS-COV-2 dataset [1] used in the paper to run the ensemble codes.

Download the dataset from [Kaggle](https://www.kaggle.com/plameneduardo/sarscov2-ctscan-dataset) and split it into train and validation sets in 80-20 ratio.

Required Directory Structure:
```

+-- data
|   +-- .
|   +-- train
|   +-- val
+-- sar-cov-2_csv
|   +--.
|   +-- inception.csv
|   +-- vgg11.csv
|   +-- wideresnet50-2.csv
+-- main.py
+-- probability_extraction
+-- utils_ensemble.py

```
To extract the probabilities on the validation set using the different models run `probability_extraction.py` and save the files in a folder. As an example the probabilities extracted on the SARS-COV-2 dataset has been saved in the folder named `sars-cov-2_csv/`.

Next, to run the ensemble model on the base learners run the following:

`python main.py --data_directory "sars-cov-2_csv/"`

References:

[1] Soares, E. & Angelov, P. A large dataset of real patients CT scans for COVID-19 identification.Harv. DataverseDOI:10.7910/DVN/SZDUQX (2020).
