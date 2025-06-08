import pandas as pd
from ucimlrepo import fetch_ucirepo

glioma_grading_clinical_and_mutation_features = fetch_ucirepo(id=759)
X_initial = glioma_grading_clinical_and_mutation_features.data.features.copy()
y_initial = glioma_grading_clinical_and_mutation_features.data.targets['Grade'].copy()

age_stats = X_initial['Age_at_diagnosis'].agg(['mean', 'median', 'min', 'max', 'std'])
age_stats['skew'] = X_initial['Age_at_diagnosis'].skew()
age_stats_df = pd.DataFrame(age_stats).transpose()
age_stats_df.rename(columns={
    'mean': 'Średnia', 'median': 'Mediana', 'min': 'Minimum',
    'max': 'Maksimum', 'std': 'Odch. stand.', 'skew': 'Skośność'
}, inplace=True)
age_stats_df.insert(0, 'Zmienna', 'Wiek w momencie diagnozy')

latex_age_table = age_stats_df.to_latex(
    index=False,
    float_format="%.2f",
    caption="Statystyki opisowe dla zmiennej numerycznej 'Wiek w momencie diagnozy'.",
    label="tab:age_stats",
    position="H"
)
print("--- Kod LaTeX dla tabeli statystyk 'Age_at_diagnosis' ---")
print(latex_age_table)

grade_counts = y_initial.value_counts().reset_index()
grade_counts.columns = ['Klasa (Grade)', 'Liczebność']
grade_counts[r'Proporcja (%)'] = (grade_counts['Liczebność'] / grade_counts['Liczebność'].sum()) * 100
grade_counts['Opis'] = grade_counts['Klasa (Grade)'].map({0: 'LGG (Glejak o niższym stopniu złośliwości)', 1: 'GBM (Glejak wielopostaciowy)'})
grade_counts = grade_counts[['Klasa (Grade)', 'Opis', 'Liczebność', r'Proporcja (%)']]

latex_grade_table = grade_counts.to_latex(
    index=False,
    float_format="%.2f",
    caption="Rozkład klas w zmiennej docelowej 'Grade'.",
    label="tab:grade_distribution",
    position="H"
)
print("\n\n--- Kod LaTeX dla tabeli rozkładu klas 'Grade' ---")
print(latex_grade_table)