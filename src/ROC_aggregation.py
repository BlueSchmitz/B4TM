import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Define paths and setup
base_path = "./results"
output_dir = os.path.join(base_path, "aggregated_roc_curves")
os.makedirs(output_dir, exist_ok=True)

models = ["baseline", "stat_test", "linear_regularization", "nonlinear_regularization"]
classes = ["HER2+", "HR+", "Triple Neg"]
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
n_points = 100  # Interpolation points

# For each class, make one plot
for class_name in classes:
    plt.figure(figsize=(7, 6))
    mean_fpr = np.linspace(0, 1, n_points)

    for model_idx, model in enumerate(models):
        model_path = os.path.join(base_path, model, "roc_data")
        files = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith(".json")]

        tprs = []
        auc_scores = []  # Precomputed AUCs from the JSON

        for file in files:
            with open(file, 'r') as f:
                data = json.load(f)
                for entry in data:
                    if entry["class"] == class_name:
                        fpr = np.array(entry["fpr"])
                        tpr = np.array(entry["tpr"])
                        interp_tpr = np.interp(mean_fpr, fpr, tpr)
                        tprs.append(interp_tpr)
                        auc_scores.append(entry["auc"])

        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = np.mean(auc_scores)
        std_auc = np.std(auc_scores)

        # Plot curve
        plt.plot(mean_fpr, mean_tpr,
                 label=f"{model} (AUC = {mean_auc:.2f} ± {std_auc:.2f})",
                 color=colors[model_idx])

    # Plot formatting
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label="Random")
    plt.title(f"ROC Curve - {class_name}", fontsize=16)
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    save_path = os.path.join(output_dir, f"{class_name.replace(' ', '_')}_2.png")
    plt.savefig(save_path, dpi=300)
    print(f" Saved: {save_path}")
    plt.close()
