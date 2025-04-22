### Plan ###
# 1. Perform pairwise t-tests between subtypes for each feature (one vs. all)
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
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

# 1. Perform pairwise t-tests between subtypes for each feature (one vs. all)
def ttest_feature_selection(data, k, results_dir='../results/feature_selection', fold_num = 0):
# the fold number is used in the file names so the files do not get overwritten! Pass it to the function!
    '''
    Feature selection based on t-tests between subtypes (one vs rest). Top k features are selected.
    Input:
        data: DataFrame containing the CNV data with 'Sample' and 'Subgroup' columns
        k: Number of top features to select for each subtype
        results_dir: Directory to save the results
        fold_num: CV fold number for saving results (to avoid overwriting files)
    Output:
        selected_data: DataFrame with selected features and their CNV values
        selected_df: DataFrame with selected features and their p-values
    '''
    # Prepare data and assign groups 
    subgroups = ['HER2+', 'HR+', 'Triple Neg']
    all_features_selected = []
    all_pvalues_selected = []
    feature_cols = data.columns[1:-1] # exclude 'Sample' and 'Subgroup' columns
    # loop through each subgroup
    for i in subgroups:
        group1 = data[data['Subgroup'] == i]
        group2 = data[data['Subgroup'] != i]
        # Perform t-test for each feature
        p_values = []
        for feature in feature_cols:
            contingency_table = pd.crosstab(data[feature], data['Subgroup'] == i)
            try:
                chi2, p_val, _, _ = chi2_contingency(contingency_table)
            except ValueError:
                p_val = 1.0  # fallback if table can't be used (e.g., constant value)
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
        selected_df.to_csv(os.path.join(results_dir, f"selected_features_{i.replace(' ', '_')}_fold_{fold_num}.csv"), index=False)

        # Keep track of selected features and their p-values
        all_features_selected.extend(top_features)
        all_pvalues_selected.extend(top_pvals)

    # Create a dataframe for the combined selected features and their p-values and save to CSV
    selected_df = pd.DataFrame({
        'Feature': all_features_selected,
        'p-value': all_pvalues_selected
    })
    selected_df.to_csv(os.path.join(results_dir, f"selected_features_with_pvalues_fold_{fold_num}.csv"), index=False)

    # Combine features of all subtypes, remove duplicates and add values back to the dataframe
    unique_features = list(set(all_features_selected)) # remove duplicates
    selected_data = data[['Sample', 'Subgroup'] + unique_features]
    selected_data.to_csv(os.path.join(results_dir, f"selected_data_fold_{fold_num}.csv"), index=False)

    return selected_data, selected_df

def remove_highly_correlated_features(selected_data, selected_df, l, results_dir='../results/feature_selection', fold_num = 0): # l is the correlation threshold that is treated as a hyperparameter 
    '''
    Remove highly correlated features based on spearman's correlation and a correlation threshold.
    Input:
        selected_data: DataFrame containing the selected features (t-test) with their CNV data and 'Sample' and 'Subgroup' columns
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
    
    # Save the dropped features and correlation
    drop_df = pd.DataFrame(drop_log, columns=['Feature_to_drop', 'Retained_feature', 'Correlation', 'p_to_drop', 'p_retained'])
    drop_df.to_csv(os.path.join(results_dir, f"correlated_features_dropped_fold_{fold_num}.csv"), index=False)
    
    # Drop features from selected features 
    data_cleaned = selected_data.drop(columns=list(features_to_drop))
    data_cleaned.to_csv(os.path.join(results_dir, f"selected_data_cleaned_fold_{fold_num}.csv"), index=False)

    return data_cleaned