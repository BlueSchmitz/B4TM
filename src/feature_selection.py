### Plan ###
# 1. Perform Kruskal-Wallis H-test across all subtypes.
    # Bootstrapping, Bonfferroni correction!!
# 2. Sort list by p-value
# 3. Select top k features for each subtype
# 4. Create a new dataframe with selected features and their p-values and report it to an output file
# 5. Create a new dataframe with the selected features and their array values to return to the main function

# Import necessary libraries
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# 1. Perform pairwise Kruskal-Wallis between subtypes for each feature
def stat_test_feature_selection(data, k, results_dir='../results/feature_selection', fold_num = 0):
# the fold number is used in the file names so the files do not get overwritten! Pass it to the function!
    '''
    Feature selection using Kruskal-Wallis H-test across all subtypes.
    Top k features with lowest FDR-adjusted p-values are selected.

    Input:
        data: DataFrame containing the CNV data with 'Sample' and 'Subgroup' columns
        k: Number of top features to select
        results_dir: Directory to save the results
        fold_num: CV fold number for saving results (to avoid overwriting files)
    Output:
        selected_data: DataFrame with selected features and their CNV values
        selected_df: DataFrame with selected features and their p-values
    '''
    # Prepare data and assign groups 
    subgroups = ['HER2+', 'HR+', 'Triple Neg']
    feature_cols = data.columns[1:-1] # exclude 'Sample' and 'Subgroup' columns
    # Perform stat test for each feature
    p_values = []
    for feature in feature_cols:
        try:
            group_values = [data[data['Subgroup'] == group][feature].values for group in subgroups]
            stat, p_val = kruskal(*group_values)
        except ValueError:
            p_val = 1.0  # fallback in case of constant values or insufficient variance
        p_values.append(p_val)

    p_values = np.array(p_values)
    # Apply FDR correction
    _, pvals_fdr, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        
    # Sort by p-value and select top k features 
    sorted_indices = np.argsort(pvals_fdr)
    top_features = feature_cols[sorted_indices][:k]
    top_pvals = pvals_fdr[sorted_indices][:k]

    # Save to CSV
    selected_df = pd.DataFrame({'Feature': top_features, 'p-value': top_pvals})
    selected_df.to_csv(os.path.join(results_dir, f"selected_features_fold_{fold_num}.csv"), index=False)

    # Dataframe to return to inner loop
    selected_data = data[['Sample', 'Subgroup'] + list(top_features)]
    selected_data.to_csv(os.path.join(results_dir, f"selected_data_fold_{fold_num}.csv"), index=False)

    return selected_data, selected_df

def nonlinear_feature_selection(data, k, alpha, results_dir='../results/feature_selection', fold_num = 0):
# the fold number is used in the file names so the files do not get overwritten! Pass it to the function!
    '''
    Feature selection using XGBoost feature importance (nonlinear).
    Top k features are selected.

    Input:
        data: DataFrame containing the CNV data with 'Sample' and 'Subgroup' columns
        k: Number of top features to select
        results_dir: Directory to save the results
        fold_num: CV fold number for saving results (to avoid overwriting files)
    Output:
        selected_data: DataFrame with selected features and their CNV values
        top_features: DataFrame with selected features and their p-values
    '''
        # Extract features and labels
    X = data.iloc[:, 1:-1]  # exclude 'Sample' and 'Subgroup'
    y = data['Subgroup']

    # Encode target labels to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train XGBoost classifier
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, alpha=alpha)
    model.fit(X, y_encoded)

    # Get feature importances
    importances = model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    # Sort and select top k
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    top_features = importance_df.head(k)

    # Save selected features
    top_features.to_csv(os.path.join(results_dir, f"xgb_selected_features_fold_{fold_num}.csv"), index=False)

    # Subset original data to include selected features
    selected_data = data[['Sample', 'Subgroup'] + list(top_features['Feature'])]
    selected_data.to_csv(os.path.join(results_dir, f"xgb_selected_data_fold_{fold_num}.csv"), index=False)

    return selected_data, top_features

def linear_feature_selection(data, k, alpha, results_dir='../results/feature_selection', fold_num = 0):
# the fold number is used in the file names so the files do not get overwritten! Pass it to the function!
    '''
    Feature selection using logistic regression feature importance (linear) with L1 regularization.
    Top k features are selected.

    Input:
        data: DataFrame containing the CNV data with 'Sample' and 'Subgroup' columns
        k/3: Number of top features to select for each subtype
        results_dir: Directory to save the results
        fold_num: CV fold number for saving results (to avoid overwriting files)
        alpha: Regularization strength for L1 penalty
    Output:
        selected_data: DataFrame with selected features and their CNV values
        importance_df: DataFrame with selected features and their p-values
    '''
    X = data.iloc[:, 1:-1]  # Exclude 'Sample' and 'Subgroup'
    y = data['Subgroup']

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_labels = le.classes_

    # Fit L1 logistic regression
    model = LogisticRegression(
        penalty='l1',
        solver='saga',
        multi_class='multinomial',
        max_iter=10000,
        C=1/alpha,
        random_state=42
    )
    model.fit(X, y_encoded)

    feature_names = np.array(X.columns)
    coef_matrix = np.abs(model.coef_)  # shape: (n_classes, n_features)

    # Select top k/3 features per class
    k_per_class = max(1, k // len(class_labels))
    selected_features = set()
    importance_records = []

    for class_idx, class_name in enumerate(class_labels):
        coefs = coef_matrix[class_idx]
        top_indices = np.argsort(coefs)[-k_per_class:]
        for idx in top_indices:
            feature = feature_names[idx]
            selected_features.add(feature)
            importance_records.append({
                'Class': class_name,
                'Feature': feature,
                'Importance': coefs[idx]
            })

    # Create result dataframe
    importance_df = pd.DataFrame(importance_records).drop_duplicates(subset='Feature')
    importance_df.to_csv(os.path.join(results_dir, f"logreg_selected_features_fold_{fold_num}.csv"), index=False)

    # Create final selected dataset
    selected_features = list(importance_df['Feature'].unique())
    selected_data = data[['Sample', 'Subgroup'] + selected_features]
    selected_data.to_csv(os.path.join(results_dir, f"logreg_selected_data_fold_{fold_num}.csv"), index=False)

    return selected_data, importance_df

def remove_highly_correlated_features(selected_data, selected_df, model_selection, l, results_dir='../results/feature_selection', fold_num = 0): # l is the correlation threshold that is treated as a hyperparameter 
    '''
    Remove highly correlated features based on spearman's correlation and a correlation threshold.
    Input:
        selected_data: DataFrame containing the selected features (stat test) with their CNV data and 'Sample' and 'Subgroup' columns
        selected_df: DataFrame containing the selected features and their p-values
        l: Correlation threshold for removing highly correlated features
    Output:
        data_cleaned: DataFrame with selected features and their CNV values after removing highly correlated features
        features_to_drop: List of features that were removed due to high correlation 
    '''
    # Compute the correlation matrix only for feature columns
    feature_cols = [col for col in selected_data.columns if col not in ['Sample', 'Subgroup']]
    corr_matrix = selected_data[feature_cols].corr(method='spearman').abs() # spearman's correlation for ordinal variables
    # Lists to keep track of features to drop
    features_to_drop = set()
    drop_log = [] # log of dropped features and their correlation values
    # Loop through upper triangle of the matrix
    for i in range(corr_matrix.shape[0]): # all rows
        for j in range(i + 1, corr_matrix.shape[1]): # only half of the columns (after diagonal)
            corr_val = corr_matrix.iloc[i, j] # get correlation value
            if corr_val >= l: # if correlation is above the threshold
                # Get feature names
                f1 = corr_matrix.index[i]
                f2 = corr_matrix.columns[j]
                if model_selection == 'stat_test':
                    # Get p-values from selected_df
                    p1_val = selected_df[selected_df['Feature'] == f1]['p-value'].values
                    p2_val = selected_df[selected_df['Feature'] == f2]['p-value'].values
                    if len(p1_val) == 0 or len(p2_val) == 0:
                        continue  # skip if p-value missing for any feature
                    p1 = p1_val[0]
                    p2 = p2_val[0]
                    # Drop the one with higher p-value
                    if p1 > p2:
                        features_to_drop.add(f1)
                        drop_log.append((f1, f2, corr_val, p1, p2))
                    else:
                        features_to_drop.add(f2)
                        drop_log.append((f2, f1, corr_val, p2, p1))
                elif model_selection in ['linear_regularization', 'nonlinear_regularization']:
                    # Get coefficients from selected_df
                    c1_val = selected_df[selected_df['Feature'] == f1]['Importance'].values
                    c2_val = selected_df[selected_df['Feature'] == f2]['Importance'].values
                    if len(c1_val) == 0 or len(c2_val) == 0:
                        continue  # skip if coefficients missing for any feature
                    c1 = abs(c1_val[0])
                    c2 = abs(c2_val[0])
                    # Drop the one with higher p-value
                    if c1 < c2:
                        features_to_drop.add(f1)
                        drop_log.append((f1, f2, corr_val, c1, c2))
                    else:
                        features_to_drop.add(f2)
                        drop_log.append((f2, f1, corr_val, c2, c1))

    # Save the dropped features and correlation
    if model_selection == 'stat_test':
        drop_df = pd.DataFrame(drop_log, columns=['Feature_to_drop', 'Retained_feature', 'Correlation', 'p_to_drop', 'p_retained'])
    elif model_selection in ['linear_regularization', 'nonlinear_regularization']:
        drop_df = pd.DataFrame(drop_log, columns=['Feature_to_drop', 'Retained_feature', 'Correlation', 'c_to_drop', 'c_retained'])
    drop_df.to_csv(os.path.join(results_dir, f"correlated_features_dropped_fold_{fold_num}.csv"), index=False)
    
    # Drop features from selected features 
    data_cleaned = selected_data.drop(columns=list(features_to_drop))
    data_cleaned.to_csv(os.path.join(results_dir, f"selected_data_cleaned_fold_{fold_num}.csv"), index=False)

    return data_cleaned