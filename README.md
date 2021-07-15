# COVID-Detection-Gompertz-Function-Ensemble
This is the official implementation of the paper titled ["Fuzzy Rank-based Fusion of CNN Models using Gompertz Function for Screening COVID-19 CT-Scans"](https://doi.org/10.1038/s41598-021-93658-y) published in "Nature- Scientific Reports".

<img src="/overall.png" style="margin: 10px;">

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
|   +-- .
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

#### References:
[1] Soares, E., Angelov, P., Biaso, S., Froes, M. H. & Abe, D. K. Sars-cov-2 ct-scan dataset: A large dataset of real patients ct scans for sars-cov-2 identification. medRxiv (2020).

# Citation
If you find this repository useful, please cite our work as follows:
```
@article{manna2021cervical,
  author={Manna, Ankur and Kundu, Rohit and Kaplun, Dmitrii and Sinitca, Aleksandr and Sarkar, Ram},
  title={A fuzzy rank-based ensemble of CNN models for classification of cervical cytology},
  journal={Scientific Reports},
  year={2021},
  month={Jul},
  day={15},
  volume={11},
  number={1},
  pages={14538},
  issn={2045-2322},
  doi={10.1038/s41598-021-93783-8},
  url={https://doi.org/10.1038/s41598-021-93783-8}
}
```
