### PLAN ###
# 1. Run 5-fold stratified CV on the data
# 2. For each fold, split the data into training and validation sets
# 3. Define hyperparameter to test during grid search
# 4. Based on the model selection (stat_test/regularization/baseline), perform feature selection and hyperparameter tuning on the training set
# 5. Evaluate the model on the validation set and record the results
# 6. Save the results to a CSV file
# 7. Return the best hyperparameters and feature selection parameters to the main function

# libraries
import numpy as np
import random
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from src.feature_selection import stat_test_feature_selection, remove_highly_correlated_features, nonlinear_feature_selection, linear_feature_selection
from sklearn.ensemble import RandomForestClassifier

def inner_loop(data, model_selection, results_dir = '../results/inner_results', outer_fold = 0):
    '''
    Inner loop function for cross-validation and hyperparameter tuning.
    Input:
        data: DataFrame containing the CNV data with 'Sample' (X) and 'Subgroup' (y) columns
        model_selection: Model to be trained (e.g., RF with feature selection or regularization, baseline)
        results_dir: Directory to save the results
        fold_num: CV outer fold number for saving results (to avoid overwriting files)
    Output:
        best_params: Dictionary of best hyperparameters for the chosen model
    '''
    # Parametergrid
    param_grid = {
        # RF hyperparameters
        'n_estimators': [50, 100],
        'max_depth': [6, 10],
        'max_features': ['sqrt'], 
        # feature selection parameters
        'k': [50, 75, 100] if model_selection == 'stat_test' else [None], # top k features
        'l': [0.6, 0.7, 0.8] if model_selection == 'stat_test' else [None], # correlation threshold 
        'alpha': [4, 7, 10] if model_selection in ['linear_regularization', 'nonlinear_regularization'] else [None] # regularization parameter for Lasso
    }

    # data
    y = data['Subgroup']
    X = data.drop(columns=['Sample', 'Subgroup'])

    # make sure folders exist
    os.makedirs(f"{results_dir}/{model_selection}/inner_results", exist_ok=True)
    os.makedirs(f"{results_dir}/{model_selection}/feature_selection", exist_ok=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_results = []
    best_params = []
    best_avg_score = 0
    param_num = 0 # to trach parameter combinations

    for params in ParameterGrid(param_grid):
        fold_scores = []
        param_num += 1 # increases with every parameter combination
        print(f"Testing parameter combination {param_num}/{len(ParameterGrid(param_grid))}...")
        print(f"Parameters: {params}")
        inner_fold = 0 # reset fold number for each parameter combination
        
        for train_idx, val_idx in skf.split(X, y):
            train_data = data.iloc[train_idx]
            val_data = data.iloc[val_idx]
            inner_fold += 1 # to track the inner fold number for each parameter combination
            print(f"Inner fold {inner_fold}...")

            # 1. Feature Selection
            if model_selection == 'stat_test':
                selected_train, selected_df = stat_test_feature_selection(
                    data=train_data, 
                    k=params['k'], 
                    results_dir=os.path.join(results_dir, model_selection, "feature_selection"), 
                    fold_num=f"{outer_fold}_{param_num}_{inner_fold}"
                )
                print("Feature selection done.")

                train_cleaned = remove_highly_correlated_features(
                    selected_data=selected_train, 
                    selected_df=selected_df, 
                    model_selection=model_selection,
                    l=params['l'], 
                    results_dir=os.path.join(results_dir, model_selection, "feature_selection"), 
                    fold_num=f"{outer_fold}_{param_num}_{inner_fold}"
                )
                print("Removed highly correlated features.")

            elif model_selection == 'baseline':
                # For baseline, we can skip feature selection
                train_cleaned = train_data.copy()
                print("Using all features without selection.")
            
            elif model_selection == 'nonlinear_regularization':
                selected_train, selected_df = nonlinear_feature_selection(
                    data=train_data, 
                    results_dir=os.path.join(results_dir, model_selection, "feature_selection"), 
                    fold_num=f"{outer_fold}_{param_num}_{inner_fold}",
                    alpha=params['alpha']
                )
                train_cleaned = selected_train.copy()
                print("Feature selection done.")

            elif model_selection == 'linear_regularization':
                selected_train, selected_df = linear_feature_selection(
                    data=train_data, 
                    results_dir=os.path.join(results_dir, model_selection, "feature_selection"), 
                    fold_num=f"{outer_fold}_{param_num}_{inner_fold}",
                    alpha=params['alpha']
                )
                train_cleaned = selected_train.copy()
                print("Feature selection done.")

            else:
                print("Invalid model selection. Choose 'stat_test', 'linear_regularization', 'nonlinear_regularization' or 'baseline'.")
                continue 

            # 2. Match features in val set
            selected_features = [col for col in train_cleaned.columns if col not in ['Sample', 'Subgroup']]
            X_train = train_cleaned[selected_features]
            y_train = train_cleaned['Subgroup']
            X_val = val_data[selected_features]
            y_val = val_data['Subgroup']

            # 3. Train and Evaluate RF
            model = RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                max_features=params['max_features'],
                random_state=42
            )

            model.fit(X_train, y_train)
            val_preds = model.predict(X_val)
            val_probs = model.predict_proba(X_val)
            train_preds = model.predict(X_train)
            train_probs = model.predict_proba(X_train)

            # 4. Log results
            all_results.append({
                'fold': f"{outer_fold}_{param_num}_{inner_fold}",
                'n_estimators': params['n_estimators'],
                'max_depth': params['max_depth'],
                'max_features': params['max_features'],
                'k': params['k'],
                'l': params['l'],
                'alpha': params['alpha'],
                'accuracy': accuracy_score(y_val, val_preds),
                'f1_weighted': f1_score(y_val, val_preds, average='weighted'),
                'roc_auc_ovr': roc_auc_score(y_val, val_probs, multi_class='ovr', average='weighted'),
                'train_accuracy': accuracy_score(y_train, train_preds),
                'train_f1_weighted': f1_score(y_train, train_preds, average='weighted'),
                'train_roc_auc_ovr': roc_auc_score(y_train, train_probs, multi_class='ovr', average='weighted')
            })

            # find out best params by average accuracy
            fold_scores.append(accuracy_score(y_val, val_preds))
        avg_score = np.mean(fold_scores)
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            best_params = {
                'fold': f"{outer_fold}_{param_num}_{inner_fold}",
                'n_estimators': params['n_estimators'],
                'max_depth': params['max_depth'],
                'max_features': params['max_features'],
                'k': params['k'],
                'l': params['l'],
                'alpha': params['alpha'],
                'accuracy': avg_score
            }

    results_df = pd.DataFrame(all_results)
    inner_results_path = os.path.join(results_dir, model_selection, "inner_results", "inner_results.csv")
    if os.path.exists(inner_results_path):
        results_df.to_csv(inner_results_path, mode='a', index=False, header=False)
    else:
        results_df.to_csv(inner_results_path, index=False)
    print(f"Saved inner results to {inner_results_path}")

    best_params_df = pd.DataFrame([best_params])
    best_params_path = os.path.join(results_dir, model_selection, "inner_results", "best_params.csv")
    if os.path.exists(best_params_path):
        best_params_df.to_csv(best_params_path, mode='a', index=False, header=False)
    else:
        best_params_df.to_csv(best_params_path, index=False)
    print(f"Saved best parameters to {best_params_path}")
    return best_params


