#!/usr/bin/env python3
### Plan ###
# 1. Run 5-fold stratified CV on the data
# 2. For each fold, call the inner loop function to perform feature selection and model training
# 3. Use returned best hyperparameters and feature selection parameters to train the model on the entire train data
# 4. Evaluate the model on the test set and record the results (save to CSV)

# Import necessary libraries
import pandas as pd
import shap
import numpy as np
from sklearn.model_selection import StratifiedKFold
import argparse
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from joblib import dump
from src.inner_loop import inner_loop
from src.feature_selection import stat_test_feature_selection, remove_highly_correlated_features, nonlinear_feature_selection, linear_feature_selection

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
        if model_selection == 'stat_test':
            selected_train, selected_df = stat_test_feature_selection(
                data=train_val_data, 
                k=best_params['k'], 
                results_dir=os.path.join(results_dir, model_selection, "outer_results"), 
                fold_num=outer_fold
            )
            # Remove highly correlated features if specified
            if best_params['l'] is not None:
                data_cleaned = remove_highly_correlated_features(
                    selected_data=selected_train,
                    selected_df=selected_df,
                    model_selection=model_selection,
                    l=best_params['l'],
                    results_dir=os.path.join(results_dir, model_selection, "outer_results"),
                    fold_num=outer_fold
                )
            else:
                data_cleaned = selected_train.copy()

        elif model_selection == 'nonlinear_regularization':
            selected_train, selected_df = nonlinear_feature_selection(
                data=train_val_data, 
                results_dir=os.path.join(results_dir, model_selection, "outer_results"), 
                fold_num=outer_fold,
                alpha=best_params['alpha']
            )
            data_cleaned = selected_train.copy()

        elif model_selection == 'linear_regularization':
            selected_train, selected_df = linear_feature_selection(
                data=train_val_data, 
                results_dir=os.path.join(results_dir, model_selection, "outer_results"), 
                fold_num=outer_fold,
                alpha=best_params['alpha']
            )
            data_cleaned = selected_train.copy()
        elif model_selection == 'baseline':
            # Use all features without selection
            data_cleaned = train_val_data.copy()
        
        # Use best hyperparameters to train final model on train+val data
        selected_features = [col for col in data_cleaned.columns if col not in ['Sample', 'Subgroup']]

        model = RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            max_features=best_params['max_features'],
            random_state=42,
            class_weight={'HER2+': 1, 'HR+': 2, 'Triple Neg': 2} # to add more importance to the other 2 classes 
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
        os.makedirs(os.path.join(results_dir, model_selection, 'models'), exist_ok=True)
        model_path = os.path.join(results_dir, model_selection, f"models/rf_model_fold{outer_fold}.joblib")
        dump(model, model_path)
        print(f"Saved model to {model_path}")

        # Multiclass ROC
        # Binarize labels (one-hot encode)
        classes = np.unique(y)
        y_test_bin = label_binarize(y_test, classes=classes)
        n_classes = y_test_bin.shape[1]
        roc_data = []
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], test_probs[:, i])
            roc_auc = auc(fpr, tpr)
            roc_data.append({
                'class': classes[i],
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': roc_auc,
                'fold': outer_fold
            })
        # Save ROC data to JSON per model
        import json
        roc_dir = os.path.join(results_dir, model_selection, "roc_data")
        os.makedirs(roc_dir, exist_ok=True)
        with open(os.path.join(roc_dir, f"roc_fold{outer_fold}.json"), "w") as f:
            json.dump(roc_data, f)

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
        
        # Save confusion matrix for each outer fold
        conf_matrix = confusion_matrix(y_test, test_preds)
        # overall confusion matrix
        if outer_fold == 1:
            total_conf_matrix = conf_matrix
        else:
            total_conf_matrix += conf_matrix
        # Plot conf
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for Outer Fold {outer_fold}')
        # Save conf as png
        conf_matrix_image_path = os.path.join(results_dir, model_selection, f"outer_results/confusion_matrix_fold{outer_fold}.png")
        plt.savefig(conf_matrix_image_path)
        plt.close()
        print(f"Saved confusion matrix for fold {outer_fold} as an image at {conf_matrix_image_path}")

        # Feature importance plot
        importances = model.feature_importances_
        features = selected_features
        # Store for aggregation later
        feature_importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances,
            'Fold': outer_fold
        })
        # Save per fold
        importance_path = os.path.join(results_dir, model_selection, f"outer_results/feature_importance_fold{outer_fold}.csv")
        feature_importance_df.to_csv(importance_path, index=False)

    # Outer loop for aggregation of results  
    # Plot aggregated confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(total_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Aggregated Confusion Matrix Across Folds')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, model_selection, "outer_results", "aggregated_confusion_matrix.png"))
    plt.close()

    # Plot aggregated feature importance
    # Aggregate feature importances across folds
    all_importances = []
    for fold in range(1, outer_fold + 1):
        path = os.path.join(results_dir, model_selection, f"outer_results/feature_importance_fold{fold}.csv")
        df = pd.read_csv(path)
        all_importances.append(df)
    combined_df = pd.concat(all_importances)
    mean_importance = combined_df.groupby('Feature')['Importance'].mean().reset_index()
    top_features = mean_importance.sort_values(by='Importance', ascending=False).head(20)
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_features, x='Importance', y='Feature', palette='viridis')
    plt.title("Top 20 Most Important Features (Avg. over folds)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, model_selection, "outer_results", "mean_feature_importance.png"))
    plt.close()

    # Plot aggregated ROC curves
    roc_dir = os.path.join(results_dir, model_selection, "roc_data")
    all_fpr = {class_name: [] for class_name in classes}  # To store FPR for each class
    all_tpr = {class_name: [] for class_name in classes}  # To store TPR for each class
    all_auc = {class_name: [] for class_name in classes}  # To store AUC for each class
    # Iterate through the saved ROC data for all folds
    for outer_fold in range(1, 6):  # Adjust num_outer_folds as needed
        roc_file = os.path.join(roc_dir, f"roc_fold{outer_fold}.json")
        if os.path.exists(roc_file):
            with open(roc_file, 'r') as f:
                roc_data = json.load(f)
            # Collect the ROC data for each class
            for class_data in roc_data:
                class_name = class_data['class']
                fpr = np.array(class_data['fpr'])
                tpr = np.array(class_data['tpr'])
                auc_value = class_data['auc']
                # Append the FPR, TPR, and AUC values for this class
                all_fpr[class_name].append(fpr)
                all_tpr[class_name].append(tpr)
                all_auc[class_name].append(auc_value)
    # Calculate the aggregated ROC curve for each class
    mean_fpr = np.linspace(0, 1, 100)  # Common FPR for interpolation
    plt.figure(figsize=(8, 6))  # Initialize the plot for all classes
    for class_name in classes:
        mean_tpr = np.zeros_like(mean_fpr)
        # Interpolate TPR values to FPR for each fold
        for fpr, tpr in zip(all_fpr[class_name], all_tpr[class_name]):
            mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr /= len(all_fpr[class_name])  # Average TPR over all folds
        mean_auc = np.mean(all_auc[class_name])  # Average AUC over all folds
        # Plot the aggregated ROC curve for this class
        plt.plot(mean_fpr, mean_tpr, label=f'{class_name} Mean ROC (AUC = {mean_auc:.2f})')
    # Plot diagonal (chance line)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance')
    # Customize the plot
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Aggregated ROC Curves Across {n_outer_folds} Folds')
    plt.legend(loc='lower right')
    # Save the plot
    roc_plot_path = os.path.join(results_dir, model_selection, "outer_results", f"aggregated_roc_curve.png")
    plt.tight_layout()
    plt.savefig(roc_plot_path)
    plt.close()
    print(f"Saved aggregated ROC curve to {roc_plot_path}")

    # Save outer results to CSV
    outer_results_df = pd.DataFrame(outer_results)
    outer_results_df.to_csv(os.path.join(results_dir, model_selection, "outer_results", "outer_results.csv"), index=False)

    # Hyperparameter tuning results
    # Accuracy
    inner_metrics_path = os.path.join(results_dir, model_selection, "inner_results", "inner_results.csv")
    df = pd.read_csv(inner_metrics_path)
    # Create param_id and aggregate
    df['param_id'] = df[['n_estimators', 'max_depth', 'max_features', 'k', 'l']].astype(str).agg('_'.join, axis=1)
    metrics = ['accuracy', 'train_accuracy']
    agg = df.groupby('param_id')[metrics].agg(['mean', 'std']).reset_index()
    agg.columns = ['param_id'] + [f"{m}_{stat}" for m, stat in agg.columns.tolist()[1:]]
    # Select top N
    top_n = min(10, len(agg))
    top_params = agg.sort_values(by='accuracy_mean', ascending=False).head(top_n)
    # Prepare data
    plot_df = pd.DataFrame({
        'param_id': top_params['param_id'],
        'Validation Mean': top_params['accuracy_mean'],
        'Validation Std': top_params['accuracy_std'],
        'Training Mean': top_params['train_accuracy_mean'],
        'Training Std': top_params['train_accuracy_std']
    })
    # Plot
    x = np.arange(top_n)
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    rects1 = ax.bar(x - width/2, plot_df['Validation Mean'], width, yerr=plot_df['Validation Std'],
                    label='Validation Accuracy', capsize=4, color='cornflowerblue', edgecolor='black', linewidth=1)
    rects2 = ax.bar(x + width/2, plot_df['Training Mean'], width, yerr=plot_df['Training Std'],
                    label='Training Accuracy', capsize=4, color='mediumpurple', edgecolor='black', linewidth=1)
    # Labels
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Top {top_n} Hyperparameter Combinations: Training vs. Validation Accuracy ± Std')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['param_id'], rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.bar_label(rects1, fmt='%.2f', padding=3)
    ax.bar_label(rects2, fmt='%.2f', padding=3)
    fig.tight_layout()
    # Save
    plot_path = os.path.join(results_dir, model_selection, "inner_results", "top_hyperparams_accuracy_comparison.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # F1 Score
    # Aggregate across folds
    metrics = ['f1_weighted', 'train_f1_weighted']
    agg = df.groupby('param_id')[metrics].agg(['mean', 'std']).reset_index()
    agg.columns = ['param_id'] + [f"{m}_{stat}" for m, stat in agg.columns.tolist()[1:]]  # Flatten columns
    # Plot top N by F1
    top_n = min(10, len(agg))
    top_params = agg.sort_values(by='f1_weighted_mean', ascending=False).head(top_n)
    plot_df = pd.DataFrame({
        'param_id': top_params['param_id'],
        'Validation Mean': top_params['f1_weighted_mean'],
        'Validation Std': top_params['f1_weighted_std'],
        'Training Mean': top_params['train_f1_weighted_mean'],
        'Training Std': top_params['train_f1_weighted_std']
    })
    # Plot
    x = np.arange(top_n)
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    rects1 = ax.bar(x - width/2, plot_df['Validation Mean'], width, yerr=plot_df['Validation Std'],
                    label='Validation F1', capsize=4, color='cornflowerblue', edgecolor='black', linewidth=1)
    rects2 = ax.bar(x + width/2, plot_df['Training Mean'], width, yerr=plot_df['Training Std'],
                    label='Training F1', capsize=4, color='mediumpurple', edgecolor='black', linewidth=1)
    # Labels
    ax.set_ylabel('Weighted F1 Score')
    ax.set_title(f'Top {top_n} Hyperparameter Combinations: Training vs. Validation F1 ± Std')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['param_id'], rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.bar_label(rects1, fmt='%.2f', padding=3)
    ax.bar_label(rects2, fmt='%.2f', padding=3)
    fig.tight_layout()
    # Save
    plot_path = os.path.join(results_dir, model_selection, "inner_results", "top_hyperparams_F1_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # AUC
    # Aggregate across folds
    metrics = ['roc_auc_ovr', 'train_roc_auc_ovr']
    agg = df.groupby('param_id')[metrics].agg(['mean', 'std']).reset_index()
    agg.columns = ['param_id'] + [f"{m}_{stat}" for m, stat in agg.columns.tolist()[1:]]  # Flatten columns
    # Plot top N by F1
    top_n = min(10, len(agg))
    top_params = agg.sort_values(by='roc_auc_ovr_mean', ascending=False).head(top_n)
    plot_df = pd.DataFrame({
        'param_id': top_params['param_id'],
        'Validation Mean': top_params['roc_auc_ovr_mean'],
        'Validation Std': top_params['roc_auc_ovr_std'],
        'Training Mean': top_params['train_roc_auc_ovr_mean'],
        'Training Std': top_params['train_roc_auc_ovr_std']
    })
    # Plot
    x = np.arange(top_n)
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    rects1 = ax.bar(x - width/2, plot_df['Validation Mean'], width, yerr=plot_df['Validation Std'],
                    label='Validation AUC', capsize=4, color='cornflowerblue', edgecolor='black', linewidth=1)
    rects2 = ax.bar(x + width/2, plot_df['Training Mean'], width, yerr=plot_df['Training Std'],
                    label='Training AUC', capsize=4, color='mediumpurple', edgecolor='black', linewidth=1)
    # Labels
    ax.set_ylabel('AUC')
    ax.set_title(f'Top {top_n} Hyperparameter Combinations: Training vs. Validation AUC ± Std')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['param_id'], rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.bar_label(rects1, fmt='%.2f', padding=3)
    ax.bar_label(rects2, fmt='%.2f', padding=3)
    fig.tight_layout()
    # Save
    plot_path = os.path.join(results_dir, model_selection, "inner_results", "top_hyperparams_AUC_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    # Plot outer results (model)
    df = outer_results_df.copy()
    # Define the metrics to compare
    metrics = ['accuracy', 'f1_weighted', 'roc_auc_ovr']
    # Compute mean and std for validation and training
    rows = []
    for metric in metrics:
        val_mean = df[metric].mean()
        val_std = df[metric].std()
        train_mean = df[f"train_{metric}"].mean()
        train_std = df[f"train_{metric}"].std()
        rows.append({'Metric': metric.capitalize(), 'Set': 'Test', 'Mean': val_mean, 'Std': val_std})
        rows.append({'Metric': metric.capitalize(), 'Set': 'Training', 'Mean': train_mean, 'Std': train_std})
    plot_df = pd.DataFrame(rows)
    x = np.arange(len(metrics))  # the label locations
    width = 0.35  # the width of the bars
    # Split data for test and train
    test_df = plot_df[plot_df['Set'] == 'Test']
    train_df = plot_df[plot_df['Set'] == 'Training']
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, test_df['Mean'], width, yerr=test_df['Std'],
                    label='Test', capsize=4, color='cornflowerblue', edgecolor='black', linewidth=1)
    rects2 = ax.bar(x + width/2, train_df['Mean'], width, yerr=train_df['Std'],
                    label='Training', capsize=4, color='mediumpurple', edgecolor='black', linewidth=1)
    # Labels and formatting
    ax.set_ylabel('Score')
    ax.set_title('Outer CV Performance: Training vs. Test ± Std')
    ax.set_xticks(x)
    custom_labels = ['Accuracy', 'Weighted F1', 'AUC']
    ax.set_xticklabels(custom_labels, rotation=0)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.bar_label(rects1, fmt='%.2f', padding=3)
    ax.bar_label(rects2, fmt='%.2f', padding=3)
    fig.tight_layout()
    # Save plot
    plot_path = os.path.join(results_dir, model_selection, "outer_results", "outer_metrics_comparison.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print("Outer CV complete.")

def main():
    parser = argparse.ArgumentParser(description="Run outer cross-validation loop for model training")
    parser.add_argument("-i", "--input", required=True, help="Path to input data file (CSV)")
    parser.add_argument("-m", "--model", required=True, choices=['stat_test', 'linear_regularization' ,'nonlinear_regularization', 'baseline'], help="Model selection method")
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