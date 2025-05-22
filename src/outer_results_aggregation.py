import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define model (feature selection) names
base_path = "./results"
models = ["baseline", "stat_test", "linear_regularization", "nonlinear_regularization"]
output_dir = os.path.join(base_path, "aggregated_outer_results")
os.makedirs(output_dir, exist_ok=True)

# Metrics to include
metrics = ["accuracy", "train_accuracy"]
metric_labels = ["Test Accuracy", "Train Accuracy"]
colors = ['tab:blue', 'tab:purple']

# Collect stats
model_stats = []
for model in models:
    csv_path = os.path.join(base_path, model, "outer_results", "outer_results.csv")
    if not os.path.exists(csv_path):
        print(f"Missing: {csv_path}")
        continue
    df = pd.read_csv(csv_path)
    stats = {
        "model": model,
        "means": [df[m].mean() for m in metrics],
        "stds": [df[m].std() for m in metrics]
    }
    model_stats.append(stats)

# Prepare plot
n_models = len(model_stats)
n_metrics = len(metrics)
bar_width = 0.3
x = np.arange(n_models)

plt.figure(figsize=(10, 6))

# Plot bars
for i in range(n_metrics):
    values = [m["means"][i] for m in model_stats]
    errors = [m["stds"][i] for m in model_stats]
    positions = x + (i - 0.5) * bar_width  # center bars
    bars = plt.bar(positions, values, width=bar_width, yerr=errors, capsize=5, label=metric_labels[i], color=colors[i], edgecolor='black')
    
    # Add values above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.12,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=9
        )

# Formatting
model_names = [m["model"] for m in model_stats]
plt.xticks(x, model_names, rotation=45, ha='right')
plt.ylabel("Accuracy", fontsize=14)
plt.title("Outer CV Train and Test Accuracy by Model", fontsize=16)
plt.ylim(0, 1.4)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc = 'upper left', fontsize=12)
plt.tight_layout()

# Save plot
save_path = os.path.join(output_dir, "outer_cv_grouped_train_accuracy_2.png")
plt.savefig(save_path, dpi=300)
plt.close()
print(f"Saved: {save_path}")
