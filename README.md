# A Benchmark for baleen whale detection

This code is intended to be used to reproduce a part of the results from the paper 
"Schall, E., Kaya, I., Debusschere, E., Devos, P. & Parcerisas, C. (2023). 
Deep learning in marine bioacoustics: A benchmark for baleen whale detection. In preparation"

## How should the data be organized 
Wav data should be converted to sequenced spectrograms. 
The spectrograms data should be contained in one folder, with one subfolder per each original class (original class,
not joined class). 
In the main folder, next to the subfolders, a csv file per original class with all samples wrongly classified can 
be added (only necessary if corrected dataset wants to be used).

Inside the subfolders, the files should be: n_locationyear_class.png, where n is the detection number of that particular 
location-year combination. 
locationyear is the location and the year of the deployment and class is the label of the image. 
For example: 1_BallenyIsland2015_20Hz

## How to specify parameters
All the parameters have to be defined in a json config file. An example can be found in this repository config.json.

Config file: 
* DATA_DIR: path to the spectrograms (main folder)
* OUTPUT_DIR: path to the folder where all the results will be stored
* LOCATIONS: List of the locations to include
* CATEGORIES : List of the categories to include
* CATEGORIES_TO_JOIN: dictionary of the name of the new joined category as key and a list of all the included 
categories as a value
* SAMPLES_PER_CLASS: number of samples per class (except noise). It does not randomly select the samples but only the 
first n samples of each class. This is to avoid a non-real representation of a small dataset, usually users annotate a 
certain period. If set to "all" it will take all the available ones

* NOISE_RATIO: list of proportion of noise of the total dataset compared to the total number of calls (from 0 to 1). 
A different model will be trained using each of the specified noise rations 
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
* learning_rate: Initial learning rate
* early_stop: number of epochs to stop after no improvement
* monitoring_metric: name of the metric to monitor (right now only accuracy available, val_accuracy)
* monitoring_direction: direction of monitoring the metric. Default to 'max'
* loss_function: name of the loss function to use

* CLASS_WEIGHTS: null, 'balanced' or dictonary with a weight per class (class as the key)
* test_after_training: bool. Set to True to perfom the test directly after training. 


```json 
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
  "EPOCHS": 100,
  "early_stop": 20,
  "monitoring_metric": "val_imbalanced_metric",
  "monitoring_direction": "max",
  "loss_function": "custom_cross_entropy",
  "learning_rate": 0.001,
  "CLASS_WEIGHTS": null,
  "test_after_training": true
}
```


## How to train a model 
To train the CNN model from the paper, run the train.py script. You will then need to input the path to the 
configuration file. If you leave it blank, it will use the config.json file in the same directory. 

When running the train script, it will create an output folder with the current timestamp. 
All the running parameters will be stored there (a model_summary.txt file and a copy of the used config file) for 
reproducibility. 

The output in the selected output folder will be: 
* A folder for each fold/location excluded with
  * the corresponding model saved
  * There is a logs file with the history of the training accuracy and loss
  * A csv file with all the data used for this fold during training
  * A plot of the training loss and training accuracy evolution during training
  * A model_summary.txt file with the model architecture
* A copy of the config file used to run that output
* Results 
  * A csv with total_confusion_matrix.csv with all the confusion matrices of all the folds and noise percentages
  * A csv for each fold and noise combination of with a list of the data used to run the test
* Results plots
  * A confusion matrix plot for each fold and noise percentage to test
  * A mean confusion matrix


