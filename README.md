# The effect of unbalanced datasets on the performance of CNNs

Here the code necessary to run the figures for the paper. 


## How to run it 
All the parameters have to be defined in a config file config.json and then the main.py script can be executed


Config file: 

* VALID_SPLIT: proportion of the data to use for validation (TOTAL data, not training data)
* TEST_SPLIT: proportion of the TOTAL data to use for test. Set to a float (0 to 1) for a single test training. 
Set to int for a Stratified KFold Cross-Validation testing strategy. The number entered will be the number of folds. 
A result for each fold will be given. 
Set to "blocked" for blocked testing. Each of the locations will be left out once. A result for each run will be given. 
* BATCH_SIZE: batch size
* EPOCHS: number of epochs
* LEARNING_RATE: 1e-3
* early_stop: 10
* class_weighted: false
* model_name: name of the model to be stored
* DATA_DIR: path to the spectrograms
* OUTPUT_DIR: path to the folder where all the results will be stored
* LOCATIONS: List of the locations to include
* CATEGORIES : List of the categories to include
* SAMPLES_PER_CLASS: number of samples per class (except noise). It does not randomly select the samples but only the 
first n samples of each class. This is to avoid a non-real representation of a small dataset, usually users annotate a 
certain period. If set to "all" it will take all the available ones
* NOISE_RATIO: proportion of noise of the total dataset compared to the total number of calls

```json 
{
  "VALID_SPLIT": 0.7,
  "TEST_SPLIT": 0.7,
  "BATCH_SIZE": 32,
  "EPOCHS": 100,
  "LEARNING_RATE": 1e-3,
  "optimizer_name": "Adam",
  "threshold_results": 0.5,
  "early_stop": 10,
  "monitoring_metric": "val_prc",
  "bias_initialization": false,
  "class_weighted": false,
  "trainable": false,
  "model_name": "IdilCNN",
  "binary": true,
  "start_from_old": null,
  "DATA_DIR": "C:/Users/cleap/Documents/Data/Sound Data/small&correctedDataset",
  "OUTPUT_DIR": "//fs/shared/onderzoek/6. Marine Observation Center/Projects/Side_Projects/Acoustics/CNN_vs_noise",
  "LOCATIONS": ["BallenyIsland", "Casey", "ElephantIsland", "SKerguelenPlateau", "Greenwich", "MaudRise"],
  "CATEGORIES" : ["20Hz20Plus", "ABZ", "DDswp", "Noise"],
  "SAMPLES_PER_CLASS": 25000,
  "NOISE_RATIO": 0.8
}
```

When running the main script, it will create an output folder with the current timestamp. 
All the running parameters will be stored there (a model_summary.txt file and a copy of the used config file) for 
reproducibility. 

