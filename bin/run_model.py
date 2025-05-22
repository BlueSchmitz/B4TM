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
import csv
# Start your coding

# import the library you need here
import pickle
import pandas as pd
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

    # suggested steps
    # Step 1: load the model from the model file
    with open(args.model_file, 'rb') as f:
        model = pickle.load(f)

    # Step 2: apply the model to the input file to do the prediction
    input_data = pd.read_csv(args.input_file, delimiter='\t')
    input_data["Region"] = input_data["Chromosome"].astype(str) + ":" + input_data["Start"].astype(str) + "-" + input_data["End"].astype(str)
    input_data = input_data.drop(columns=["Chromosome", "Start", "End", "Nclone"])
    input_data = input_data.set_index("Region").T.reset_index()
    input_data = input_data.rename(columns={"index": "Sample"})
    final_features_path = "./results/final_model/feature_names_final_model.csv"
    final_features_df = pd.read_csv(final_features_path)
    final_features = final_features_df['Feature'].tolist()
    X = input_data[final_features]
    predictions = model.predict(X)

    # Step 3: write the prediction into the desinated output file
    output_df = pd.DataFrame({
        'Sample': input_data["Sample"],
        'Subgroup': predictions
    })

    output_df.to_csv(args.output_file, sep='\t', index=False, quoting=csv.QUOTE_ALL)

    print(f"Predictions saved to {args.output_file}")
    # End your coding


if __name__ == '__main__':
    main()