# The effect of unbalanced datasets on the performance of CNNs

Here the code necessary to run the figures for the paper. 

## How should the data be organized 
Data should be one folder, with one subfolder per each original class (not joined class). 
In the main folder, next to the subfolders, a csv file per original class with all the samples wrongly classified
(only necessary if corrected dataset wants to be used)

Inside the subfolders, the files should be: n_locationyear_class, where n is the detection number of that particular 
location-year combination. locationyear is the location and the year of the deployment and class is the label of the image. 
For example: 1_BallenyIsland2015_20Hz

## How to run it 
All the parameters have to be defined in a config file config.json and then the training.py script can be executed


Config file: 
* DATA_DIR: path to the spectrograms
* OUTPUT_DIR: path to the folder where all the results will be stored
* LOCATIONS: List of the locations to include
* CATEGORIES : List of the categories to include
* CATEGORIES_TO_JOIN: dictionary of the name of the new joined category as key and a list of all the included 
categories as a value
* SAMPLES_PER_CLASS: number of samples per class (except noise). It does not randomly select the samples but only the 
first n samples of each class. This is to avoid a non-real representation of a small dataset, usually users annotate a 
certain period. If set to "all" it will take all the available ones

* NOISE_RATIO: proportion of noise of the total dataset compared to the total number of calls (from 0 to 1). It will be 
the one used in the training
* NOISE_RATIO_TEST: can be a single number (0.5) or a list [0.5, 0.6, 0.7, 0.8, 0.9]. Will be the noise ratio used 
for testing
* USE_CORRECTED_DATASET: bool, set to true to use the corrected dataset (only allows for 2500 samples per class!), 
set to false otherwise


* VALID_SPLIT: proportion of the data to use for validation (TOTAL data, not training data)
* TEST_SPLIT: proportion of the TOTAL data to use for test. Set to a float (0 to 1) for a single test training. 
Set to int for a Stratified KFold Cross-Validation testing strategy. The number entered will be the number of folds. 
A result for each fold will be given. 
Set to "blocked" for blocked testing. Each of the locations will be left out once. A result for each run will be given. 

* model_name: name of the model to be stored
* BATCH_SIZE: batch size
* EPOCHS: number of epochs
* LEARNING_RATE: 1e-3
* early_stop: 10
* monitoring_metric: name of the metric to monitor (right now only accuracy available, val_accuracy)

* CLASS_WEIGHTS: null, 'balanced' or dictonary with a weight per class (class as the key)


```json 
{
{
  "DATA_DIR": "C:/Users/cleap/Documents/Data/Sound Data/cnn_and_noise",
  "OUTPUT_DIR": "//fs/shared/onderzoek/6. Marine Observation Center/Projects/Side_Projects/Acoustics/CNN_vs_noise/",
  "LOCATIONS": ["BallenyIsland", "Casey", "ElephantIsland", "SKerguelenPlateau", "Greenwich", "MaudRise"],
  "CATEGORIES" : ["20Plus", "20Hz", "A", "B", "D", "Dswp", "Z", "Noise"],
  "CATEGORIES_TO_JOIN": {
                          "20Hz20Plus":  ["20Plus","20Hz"],
                          "ABZ":  ["A","B", "Z"]
                        },
  "SAMPLES_PER_CLASS": 250,

  "NOISE_RATIO": 0.5,
  "NOISE_RATIO_TEST": [0.5, 0.6, 0.7, 0.8]
  "USE_CORRECTED_DATASET": true,

  "VALID_SPLIT": 0.3,
  "TEST_SPLIT": 5,

  "model_name": "IdilCNN",
  "BATCH_SIZE": 32,
  "EPOCHS": 2,
  "LEARNING_RATE": 1e-3,
  "early_stop": 2,
  "monitoring_metric": "val_accuracy"
}
```

When running the training script, it will create an output folder with the current timestamp. 
All the running parameters will be stored there (a model_summary.txt file and a copy of the used config file) for 
reproducibility. 

There are 4 config files under the configs/ folder. These are the to use for each run. 
For each test, run "training.py" for each of the configs and select a different folder as an output (in the config file)

The output in the selected output folder will be: 
* A folder for each fold/location excluded with the corresponding model saved there.
  * There is a logs file in the model with the history of the accuracy and the loss
* A copy of the config file used to run that output 
* A model_summary.txt file with the model architecture
* Plots
  * A confusion matrix plot for each fold and noise percentage (if more than one)
  * A plot of the training loss and training accuracy evolution during training, per fold
* Results 
  * A csv with total_confusion_matrix.csv with all the confusion matrices of all the folds and noise percentages


