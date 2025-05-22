import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.multitest import multipletests
from scipy.stats import kruskal
import pickle

# script to retrain the model on the whole dataset using the best hyperparameters 
data_path = "./data"
data = pd.read_csv(os.path.join(data_path, 'Train_call.txt'), delimiter='\t')
labels = pd.read_csv(os.path.join(data_path, 'Train_clinical.txt'), delimiter='\t')
data["Region"] = data["Chromosome"].astype(str) + ":" + data["Start"].astype(str) + "-" + data["End"].astype(str)
data = data.drop(columns=["Chromosome", "Start", "End", "Nclone"])
data = data.set_index("Region").T.reset_index()
data = data.rename(columns={"index": "Sample"})
data = data.merge(labels, on="Sample", how="left")

X = data.drop(columns=['Sample', 'Subgroup'])
y = data['Subgroup']

results_dir = "./results/final_model"
os.makedirs(f"{results_dir}", exist_ok=True)

### Feature selection: Krukal-Wallis test ###
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
top_features = feature_cols[sorted_indices][:200]
top_pvals = pvals_fdr[sorted_indices][:200]

# Save to CSV
selected_df = pd.DataFrame({'Feature': top_features, 'p-value': top_pvals})
selected_df.to_csv(os.path.join(results_dir, f"selected_features_final_model.csv"), index=False)

selected_data = data[['Sample', 'Subgroup'] + list(top_features)]
selected_data.to_csv(os.path.join(results_dir, f"selected_data_final_model.csv"), index=False)

print(f"Selected {len(top_features)} features based on Kruskal-Wallis test with FDR correction.")

### Remove correlated features
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
        if corr_val >= 0.7: # if correlation is above the threshold
            # Get feature names
            f1 = corr_matrix.index[i]
            f2 = corr_matrix.columns[j]
            if f1 in features_to_drop or f2 in features_to_drop:
                continue # skip if already marked for dropping
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
drop_df.to_csv(os.path.join(results_dir, f"correlated_features_dropped_fold_final_model.csv"), index=False)
    
# Drop features from selected features 
data_cleaned = selected_data.drop(columns=list(features_to_drop))
data_cleaned.to_csv(os.path.join(results_dir, f"selected_data_cleaned_fold_final_model.csv"), index=False)
final_features = [col for col in data_cleaned.columns if col not in ['Sample', 'Subgroup']]

print(f"Removed {len(features_to_drop)} correlated features. Remaining features: {len(final_features)}")

# Model and hyperparameters
model = RandomForestClassifier(
n_estimators=50,
max_depth=6,
max_features='sqrt',
random_state=42
)

# Train the model
selected_features = [col for col in data_cleaned.columns if col not in ['Sample', 'Subgroup']]
X_train = data_cleaned[selected_features]
y_train = data_cleaned['Subgroup']
model.fit(X_train, y_train)

print("Model training complete.")

# Save the model
model_path = os.path.join(results_dir, "final_model.pkl")
pickle.dump(model, open(model_path, 'wb'))
# Save the feature names
feature_names_path = os.path.join(results_dir, "feature_names_final_model.csv")
pd.DataFrame({'Feature': final_features}).to_csv(feature_names_path, index=False)

print(f"Model and final feature names saved to {model_path} and {feature_names_path}.")