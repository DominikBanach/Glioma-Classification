import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

sns.set_theme(style="whitegrid")
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

glioma_grading_clinical_and_mutation_features = fetch_ucirepo(id=759)
X_initial = glioma_grading_clinical_and_mutation_features.data.features.copy()
y_initial = glioma_grading_clinical_and_mutation_features.data.targets['Grade'].copy()

df_full = pd.concat([X_initial, y_initial], axis=1)

clinical_features = ['Gender', 'Age_at_diagnosis', 'Race']
mutation_features = [col for col in X_initial.columns if col not in clinical_features]

n_features = len(mutation_features)
n_cols = 4
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4), sharey=True)
axes = axes.flatten()

for i, gene_col in enumerate(mutation_features):
    proportions = df_full.groupby(gene_col)['Grade'].value_counts(normalize=True).mul(100).rename('Proporcja (%)').reset_index()

    sns.barplot(x=gene_col, y='Proporcja (%)', hue='Grade', data=proportions, ax=axes[i], palette="viridis")
    
    axes[i].set_title(f'Proporcje klas dla: {gene_col}')
    axes[i].set_xlabel('Status (0: Niezmutowany, 1: Zmutowany)')
    axes[i].set_ylabel('Proporcja pacjentów (%)' if i % n_cols == 0 else '')
    axes[i].set_ylim(0, 100)

    if i == 0:
        handles, labels = axes[i].get_legend_handles_labels()
        grade_map = {0: 'LGG (Grade 0)', 1: 'GBM (Grade 1)'}
        labels = [grade_map.get(int(label), label) for label in labels]
        fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.98))
    
    if axes[i].get_legend() is not None:
        axes[i].get_legend().remove()

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.suptitle("Proporcjonalny Rozkład Klas 'Grade' w Zależności od Statusu Mutacji", fontsize=18, y=1.0)
plt.show()