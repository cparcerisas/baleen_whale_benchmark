# Import required modules
import json
import pandas as pd
import os
import numpy as np


def main():

    # Output file from test with the noise class listed in the last column
    raw_detections_path = "/Path/to/predictions.csv"

    # Ground truth file with the noise class listed in the last column
    test_annots_path = "/Path/to/truecoverage.csv"

    # Label list file where the order of labels is according to the columns in the predictions
    # and ground truth and noise beeing the last label in the list
    label_list_path = "/Path/to/classes_list.json"

    # Output path for results file
    output_path = "/Path/to/store/results/"

    # Read true coverage, predictions and list of labels
    f = open(label_list_path)
    label_list = json.load(f)

    labels = pd.read_csv(raw_detections_path, header=0, index_col=0)
    labels = labels.to_numpy()

    true_coverage = pd.read_csv(test_annots_path, header=0, index_col=0)
    true_coverage = true_coverage.to_numpy()

    # Create confusion matrix for segments with potentially multiple labels
    confusion_matrix = np.zeros((len(label_list) + 1, len(label_list) + 1), dtype=int)
    for true_label in range(len(label_list)):
        for predicted_label in range(len(label_list)):
            confusion_matrix[true_label, predicted_label] = sum(
                labels[(true_coverage[:, true_label] == 1) & (true_coverage[:, predicted_label] != 1), predicted_label])

        confusion_matrix[true_label, true_label] = sum(labels[true_coverage[:, true_label] == 1, true_label])
        confusion_matrix[true_label, -1] = sum(true_coverage[:,true_label])
        confusion_matrix[-1,true_label] = sum(labels[:,true_label])

    # Extract diagonal and non-diagonal from confusion matrix
    dia = np.diag(confusion_matrix)
    non_dia_ind = ~np.eye(confusion_matrix[0:-2,0:-2].shape[0],dtype=bool)
    non_dia = confusion_matrix[0:-2,0:-2]
    non_dia = non_dia[np.array(non_dia_ind)]
    non_dia = non_dia.reshape([(len(label_list) - 1), int(len(non_dia) / (len(label_list) - 1))])

    # Extract single TCRs, NMRs, and CMRs per entry in the confusion matrix
    TCRs = []
    NMRs = []
    CMRs = []
    for classes in range(len(label_list) -1):
        TCRs.append(dia[classes]/confusion_matrix[classes,-1])
        NMRs.append(confusion_matrix[-2,classes])
        for cols in range(np.shape(non_dia)[1]):
            CMRs.append(non_dia[classes,cols]/confusion_matrix[classes,-1])

    # Calculate TCR, NMR, CMR, and F value
    TCR = np.average(TCRs)
    NMR = sum(NMRs) / confusion_matrix[-2,-1]
    CMR = np.average(CMRs)
    F = np.average([TCR, (1 - NMR), (1 - NMR), CMR])

    # Save confusion matrix and evaluation metrics to csv
    DF = pd.DataFrame(confusion_matrix)
    model_predictions = os.path.splitext(os.path.splitext(os.path.basename(raw_detections_path))[0])[0]
    DF.to_csv(os.path.join(output_path,'.'.join([''.join([model_predictions, '_confusion','_NMR=',NMR.round(2).astype(str),
                                                          '_CMR=',CMR.round(2).astype(str),'_TCR=',TCR.round(2).astype(str),'_F=',F.round(2).astype(str)]),'csv'])))



if __name__ == "__main__":
    main()