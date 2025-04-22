#!/usr/bin/env python3
### Plan ###
# 1. Run 5-fold stratified CV on the data
# 2. For each fold, call the inner loop function to perform feature selection and model training
# 3. Use returned best hyperparameters and feature selection parameters to train the model on the entire train data
# 4. Evaluate the model on the test set and record the results (save to CSV)

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from joblib import dump
from src.inner_loop import inner_loop
from src.feature_selection import ttest_feature_selection, remove_highly_correlated_features

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def outer_loop(data_path, model_selection, results_dir, n_outer_folds=5):
    # Load data
    data = pd.read_csv(os.path.join(data_path, 'Train_call.txt'), delimiter='\t')
    labels = pd.read_csv(os.path.join(data_path, 'Train_clinical.txt'), delimiter='\t')
    data["Region"] = data["Chromosome"].astype(str) + ":" + data["Start"].astype(str) + "-" + data["End"].astype(str)
    data = data.drop(columns=["Chromosome", "Start", "End", "Nclone"])
    data = data.set_index("Region").T.reset_index()
    data = data.rename(columns={"index": "Sample"})
    data = data.merge(labels, on="Sample", how="left")

    X = data.drop(columns=['Sample', 'Subgroup'])
    y = data['Subgroup']

    # make sure folders exist
    os.makedirs(f"{results_dir}/{model_selection}/outer_results", exist_ok=True)

    skf_outer = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=42)

    outer_fold = 0
    outer_results = []

    for train_idx, test_idx in skf_outer.split(X, y):
        outer_fold += 1
        print(f"Outer fold {outer_fold}...")

        train_val_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]

        # Run inner CV loop to get best hyperparameters
        best_params = inner_loop(train_val_data, model_selection, results_dir=results_dir, outer_fold=outer_fold)

        # perform feature selection based on best hyperparameters 
        if model_selection == 't_test':
            selected_train, _ = ttest_feature_selection(
                train_val_data, 
                k=best_params['k'], 
                results_dir=results_dir, 
                fold_num=outer_fold
            )
            # Remove highly correlated features if specified
            if best_params['l'] is not None:
                data_cleaned = remove_highly_correlated_features(selected_train, l=best_params['l'])
            else:
                data_cleaned = selected_train.copy()

        elif model_selection == 'regularization':
            # Implement regularization feature selection here (if needed)
            pass

        elif model_selection == 'baseline':
            # Use all features without selection
            data_cleaned = train_val_data.copy()
        
        # Use best hyperparameters to train final model on train+val data
        selected_features = [col for col in data_cleaned.columns if col not in ['Sample', 'Subgroup']]

        model = RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            max_features=best_params['max_features'],
            random_state=42
        )
        X_trainval = data_cleaned[selected_features]
        y_trainval = data_cleaned['Subgroup']
        X_test = test_data[selected_features]
        y_test = test_data['Subgroup']

        model.fit(X_trainval, y_trainval)
        test_preds = model.predict(X_test)
        test_probs = model.predict_proba(X_test)
        trainval_preds = model.predict(X_trainval)
        trainval_probs = model.predict_proba(X_trainval)

        # Save model
        model_path = os.path.join(results_dir, model_selection, f"models/rf_model_fold{outer_fold}.joblib")
        dump(model, model_path)
        print(f"Saved model to {model_path}")

        # Evaluate on test set and log results 
        outer_results.append({
                'fold': outer_fold,
                'n_estimators': best_params['n_estimators'],
                'max_depth': best_params['max_depth'],
                'max_features': best_params['max_features'],
                'k': best_params['k'],
                'l': best_params['l'],
                'accuracy': accuracy_score(y_test, test_preds),
                'f1_weighted': f1_score(y_test, test_preds, average='weighted'),
                'roc_auc_ovr': roc_auc_score(y_test, test_probs, multi_class='ovr', average='weighted'),
                'train_accuracy': accuracy_score(y_trainval, trainval_preds),
                'train_f1_weighted': f1_score(y_trainval, trainval_preds, average='weighted'),
                'train_roc_auc_ovr': roc_auc_score(y_trainval, trainval_probs, multi_class='ovr', average='weighted')
            })
    outer_results_df = pd.DataFrame(outer_results)
    outer_results_df.to_csv(os.path.join(results_dir, model_selection, "outer_results", "outer_results.csv"), index=False)

    print("Outer CV complete.")

def main():
    parser = argparse.ArgumentParser(description="Run outer cross-validation loop for model training")
    parser.add_argument("-i", "--input", required=True, help="Path to input data file (CSV)")
    parser.add_argument("-m", "--model", required=True, choices=['t_test', 'regularization', 'baseline'], help="Model selection method")
    parser.add_argument("-r", "--results", required=True, help="Path to directory where results should be saved")
    parser.add_argument("-f", "--folds", required=False, help="Number of outer loop folds", type=int, default=5)
    # Parse options
    args = parser.parse_args()

    if args.input is None:
        sys.exit('Input is missing!')

    if args.results is None:
        sys.exit('Output is not designated!')
    
    if not os.path.exists(args.results):
        os.makedirs(args.results)

    outer_loop(args.input, args.model, args.results, args.folds)


if __name__ == '__main__':
    main()