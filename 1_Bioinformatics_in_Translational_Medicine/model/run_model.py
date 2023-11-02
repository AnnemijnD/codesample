#!/usr/bin/env python3
"""Reproduce your result by your saved model.

This is a script that helps reproduce your prediction results using your saved
model. This script is unfinished and you need to fill in to make this script
work. If you are using R, please use the R script template instead.

The script needs to work by typing the following commandline (file names can be
different):

python3 run_model.py -i unlabelled_sample.txt -m model.pkl -o output.txt

"""

# author: Chao (Cico) Zhang
# date: 31 Mar 2017

import argparse
import sys
# Start your coding

import pandas as pd
import pickle
import sklearn
import numpy as np

# End your coding


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Reproduce the prediction')
    parser.add_argument('-i', '--input', required=True, dest='input_file',
                        metavar='unlabelled_sample.txt', type=str,
                        help='Path of the input file')
    parser.add_argument('-m', '--model', required=True, dest='model_file',
                        metavar='model.pkl', type=str,
                        help='Path of the model file')
    parser.add_argument('-o', '--output', required=True,
                        dest='output_file', metavar='output.txt', type=str,
                        help='Path of the output file')
    # Parse options
    args = parser.parse_args()

    if args.input_file is None:
        sys.exit('Input is missing!')

    if args.model_file is None:
        sys.exit('Model file is missing!')

    if args.output_file is None:
        sys.exit('Output is not designated!')

    # Start your coding
    # Step 1: load the data
    pickled_file = pickle.load(open(args.model_file, "rb"))
    input = args.input_file
    model = pickled_file[0]
    features = pickled_file[1]
    output = args.output_file

    # process the patient data
    dfcall = pd.read_csv(input, delimiter="\t")

    #Transposes the dataframe
    temp_df = dfcall.T

    #Removes the first 4 lines corresponding to the chromosomal
    rotated_df=temp_df[4::]

    #Sets the new index based
    rotated_df = rotated_df.reset_index()

    final_df=rotated_df

    # Get the final data into numpy array
    X = final_df.iloc[:,1:2835].values
    labels = final_df.iloc[:,0].values

    # feature selection with specified features
    total_feat_N = len(X[0])
    delete_ind = []
    for i in range(total_feat_N):
        if i not in features:
            delete_ind.append(i)

    # deletes the features that can be deleted
    X = np.delete(X, delete_ind, 1)

    y_pred = model.predict(X)
    # produce dictionary to make csv file
    dict = {"Sample":[], "Subgroup":[]}

    # reassign labels of the patients
    for i in range(len(y_pred)):
        dict["Sample"].append(labels[i])
        dict["Subgroup"].append(y_pred[i])

    # save the file
    df = pd.DataFrame(dict)

    df.to_csv("../results/"+output, sep="\t", index=False, quoting=1)
    # End your coding


if __name__ == '__main__':
    main()
