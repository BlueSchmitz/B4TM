import os
import pandas as pd

# Aggregate feature importances across folds
all_importances = []
for fold in range(1, 6):
    path = f"./results/stat_test/outer_results/feature_importance_MDA_fold{fold}.csv"
    df = pd.read_csv(path)
    df['Fold'] = fold  # Track the fold number
    all_importances.append(df)

# Combine data from all folds
combined_df = pd.concat(all_importances)

# Count how many folds each feature appeared in
appearance_counts = combined_df.groupby('Feature')['Fold'].nunique().reset_index()
appearance_counts = appearance_counts.rename(columns={'Fold': 'Appearances'})

# Compute mean and standard deviation of importance per feature
importance_stats = (
    combined_df.groupby('Feature')['Importance']
    .agg(['mean', 'std'])
    .reset_index()
    .rename(columns={'mean': 'MeanImportance', 'std': 'StdImportance'})
)

# Merge appearance counts with importance stats
importance_summary = pd.merge(importance_stats, appearance_counts, on='Feature')

# Sort by mean importance
importance_summary = importance_summary.sort_values(by='MeanImportance', ascending=False)

# Save to CSV
output_path = os.path.join("results", "stat_test", "outer_results", "feature_importance_summary_MDA.csv")
importance_summary.to_csv(output_path, index=False)