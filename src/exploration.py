# load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# open data 
train = pd.read_csv('./data/Train_call.txt', delimiter='\t')
labels = pd.read_csv('./data/Train_clinical.txt', delimiter='\t')
#print(train.shape)
#print(data.head())
#print(labels.head())

## Reshape data for classification ##
train1 = train.copy().drop(columns=["Start", "End", "Nclone"]) # for chromosome CNV frequency plot
train["Region"] = train["Chromosome"].astype(str) + ":" + train["Start"].astype(str) + "-" + train["End"].astype(str)
train = train.drop(columns=["Chromosome", "Start", "End", "Nclone"])
train = train.set_index("Region").T.reset_index()
train = train.rename(columns={"index": "Sample"})

# Merge data and labels
data = train.merge(labels, on="Sample", how="left")
#print(data.head())
#print(data.shape)

# Check for missing values
missing_values = data.isnull().sum()
print("Missing values in each column:")
print(missing_values[missing_values > 0])

'''
## Class distribution ##
class_distribution = data["Subgroup"].value_counts()
print("\nClass distribution:")
print(class_distribution)
print("\nClass distribution (percentage):")
print(class_distribution / class_distribution.sum() * 100)
histogram = data["Subgroup"].value_counts().plot(kind='bar', title='Class Distribution')
plt.xlabel('Subgroup')
plt.ylabel('Count')
plt.subplots_adjust(bottom=0.25)
plt.show()
'''

'''
# CNV distribution
cnv_distribution = data.iloc[:, 1:-1].sum(axis=0)/ len(data)
histogram = cnv_distribution.plot(kind='bar', title='CNV Distribution')
plt.xlabel('Genomic Region')
plt.ylabel('Count')
plt.xticks([])
plt.show()
'''

'''
## Frequency plots ##
# Chromosome CNV frequency plot
cnv_per_chromosome = train1.groupby("Chromosome").sum().sum(axis=1)
regions_per_chromosome = train1["Chromosome"].value_counts().sort_index()
cnv_per_chromosome_normalized = cnv_per_chromosome / regions_per_chromosome
cnv_per_chromosome_normalized.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("CNV Distribution Per Chromosome")
plt.xlabel("Chromosome")
plt.ylabel("Average CNV Alterations per Region")
plt.xticks(rotation=0)
plt.show()

# Chromosome-wise CNV Frequency Plot
cnv_classes = [-1, 0, 1, 2]
cnv_counts = train1.melt(id_vars=["Chromosome"], value_vars=train1.columns[1:-1], var_name="Sample", value_name="CNV")
cnv_counts = cnv_counts.groupby(["Chromosome", "CNV"]).size().unstack(fill_value=0)

# Ensure all CNV categories are included
for cnv_class in cnv_classes:
    if cnv_class not in cnv_counts.columns:
        cnv_counts[cnv_class] = 0

# Convert counts to percentages per chromosome
cnv_percentages = cnv_counts.div(cnv_counts.sum(axis=1), axis=0) * 100

# Plot stacked bar chart
plt.figure(figsize=(12, 6))
cnv_percentages.plot(kind="bar", stacked=True, colormap="coolwarm", edgecolor="black", width=0.8)

# Formatting
plt.title("Chromosome-wise CNV Frequency Plot")
plt.xlabel("Chromosome")
plt.ylabel("Percentage of CNV Events")
plt.xticks(rotation=0)
plt.legend(title="CNV Type", labels=["Loss (-1)", "Normal (0)", "Gain (1)", "Amplification (2)"])
plt.show()
'''

## Can clusters be identified? ##
# meancenter data
X = data.iloc[:, 1:-1]
y = data["Subgroup"]
samples = data["Sample"]
X_centered = X - X.mean()

'''
# PCA with points colored by Subgroup
from sklearn.decomposition import PCA
# PCA with 2 PCs
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_centered)
# Plot PCA
plt.figure(figsize=(10, 7))
subtypes = y.unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(subtypes))) # color per subtype
for subtype, color in zip(subtypes, colors):
    idx = y == subtype
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=subtype, c=[color], alpha=0.7, edgecolor='k')
plt.title("PCA of CNV Data (Mean-Centered)")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.legend(title="Subtype")
plt.grid(True)
plt.tight_layout()
plt.show()


# t-SNE with points colored by Subgroup
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=10, random_state=42)
X_tsne = tsne.fit_transform(X_centered)
# Plot
subtypes = y.unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(subtypes))) # color per subtype
plt.figure(figsize=(8, 4))
for subtype, color in zip(subtypes, colors):
    idx = y == subtype
    plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=subtype, c=[color], alpha=0.7, edgecolor='k')

plt.title("t-SNE of CNV Data")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(title="Subtype")
plt.grid(True)
plt.tight_layout()
plt.show()

# UMAP
import umap as umap
reducer = umap.UMAP(random_state=42)
X_umap = reducer.fit_transform(X_centered)
plt.figure(figsize=(10, 7))
for subtype, color in zip(subtypes, colors):
    idx = y == subtype
    plt.scatter(X_umap[idx, 0], X_umap[idx, 1], label=subtype, c=[color], alpha=0.7, edgecolor='k')
plt.title("UMAP of CNV Data")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend(title="Subtype")
plt.grid(True)
plt.tight_layout()
plt.show()
'''