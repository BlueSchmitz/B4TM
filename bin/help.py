import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Hyperparameter tuning results
# Accuracy
inner_metrics_path = os.path.join("results", "stat_test", "inner_results", "inner_results.csv")
df = pd.read_csv(inner_metrics_path)
# Create param_id and aggregate
df['param_id'] = df[['n_estimators', 'max_depth', 'k', 'l']].astype(str).agg('_'.join, axis=1)
metrics = ['accuracy', 'train_accuracy']
agg = df.groupby('param_id')[metrics].agg(['mean', 'std']).reset_index()
agg.columns = ['param_id'] + [f"{m}_{stat}" for m, stat in agg.columns.tolist()[1:]]
# Select top N
top_n = min(20, len(agg))
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
# Show
plt.show()
# Save
#plot_path = os.path.join("./results", "linear_regularization", "inner_results", "top_hyperparams_accuracy_comparison_2.png")
#os.makedirs(os.path.dirname(plot_path), exist_ok=True)
#plt.savefig(plot_path, dpi=300)
#plt.close()