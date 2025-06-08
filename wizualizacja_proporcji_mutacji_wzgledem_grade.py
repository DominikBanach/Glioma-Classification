import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
import os

sns.set_theme(style="whitegrid")

glioma_grading_clinical_and_mutation_features = fetch_ucirepo(id=759)
X_initial = glioma_grading_clinical_and_mutation_features.data.features.copy()
y_initial = glioma_grading_clinical_and_mutation_features.data.targets['Grade'].copy()

df_full = pd.concat([X_initial, y_initial], axis=1)

clinical_features = ['Gender', 'Age_at_diagnosis', 'Race']
mutation_features = [col for col in X_initial.columns if col not in clinical_features]

output_dir = "visuals_new"
os.makedirs(output_dir, exist_ok=True)
print(f"Wykresy zostaną zapisane w folderze: '{output_dir}'")

for gene_col in mutation_features:
    fig, ax = plt.subplots(figsize=(6, 4))

    proportions = df_full.groupby(gene_col)['Grade'].value_counts(normalize=True).mul(100).rename('Proporcja (%)').reset_index()

    sns.barplot(x=gene_col, y='Proporcja (%)', hue='Grade', data=proportions, ax=ax, palette="viridis")
    
    ax.set_title(f'Rozkład klas dla: {gene_col}')
    ax.set_xlabel('Status (0: Niezmutowany, 1: Zmutowany)')
    ax.set_ylabel('Proporcja pacjentów (%)')
    ax.set_ylim(0, 100)

    handles, labels = ax.get_legend_handles_labels()
    grade_map = {0: 'LGG (Grade 0)', 1: 'GBM (Grade 1)'}
    labels = [grade_map.get(int(label), label) for label in labels]
    ax.legend(handles, labels, title='Grade', loc='upper right')
    
    output_filename = os.path.join(output_dir, f"dystrybucja_{gene_col.lower()}.png")
    
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close(fig) 

print(f"Zakończono generowanie {len(mutation_features)} wykresów.")