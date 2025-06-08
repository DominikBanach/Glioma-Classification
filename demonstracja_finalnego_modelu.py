import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from ucimlrepo import fetch_ucirepo

print("Wczytywanie i przygotowywanie danych...")
# fetch dataset
glioma_grading_clinical_and_mutation_features = fetch_ucirepo(id=759)
X_full = glioma_grading_clinical_and_mutation_features.data.features.copy()
y_full = glioma_grading_clinical_and_mutation_features.data.targets['Grade'].copy()

# Preprocessing CAŁEGO zbioru X_full
race_mapping = {
    'white': 0, 
    'black or african american': 1, 
    'asian': 2, 
    'american indian or alaska native': 3
}
X_full['Race'] = X_full['Race'].map(race_mapping).fillna(-1)

scaler_final = StandardScaler() 
age_diag_col = 'Age_at_diagnosis'
X_full[age_diag_col] = scaler_final.fit_transform(X_full[[age_diag_col]])
print("Dane przygotowane.")

final_selected_feature_names = [
    'IDH2', 'TP53', 'CIC', 'IDH1', 'SMARCA4', 'EGFR', 'NOTCH1', 
    'GRIN2A', 'Race', 'NF1', 'PTEN', 'RB1', 'MUC16', 'FUBP1', 
    'PDGFRA', 'Age_at_diagnosis', 'PIK3R1', 'ATRX'
] 

X_final_selected = X_full[final_selected_feature_names]

print(f"\nTrenowanie finalnego modelu na {X_final_selected.shape[1]} wybranych cechach i {X_final_selected.shape[0]} obserwacjach.")

final_lr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=0, solver='liblinear', max_iter=200)
final_svm = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=0)
final_rf_base = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=0, n_jobs=-1) # Zmieniono nazwę na final_rf_base
final_ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=0) 

final_lr.fit(X_final_selected, y_full)
final_svm.fit(X_final_selected, y_full)
final_rf_base.fit(X_final_selected, y_full) 
final_ada.fit(X_final_selected, y_full)

print("Finalne modele bazowe dla Ens13 wytrenowane.")


wiek_pacjent1_oryginalny = 35.0
wiek_pacjent1_znormalizowany = scaler_final.transform(np.array([[wiek_pacjent1_oryginalny]]))[0,0]

pacjent1_data_dict = {feature: 0 for feature in final_selected_feature_names} 
pacjent1_data_dict.update({
    'Age_at_diagnosis': wiek_pacjent1_znormalizowany,
    'Race': 0, 
    'IDH1': 1,
    'ATRX': 1,
    'CIC': 1,
    'TP53': 1 
})

wiek_pacjent2_oryginalny = 65.0
wiek_pacjent2_znormalizowany = scaler_final.transform(np.array([[wiek_pacjent2_oryginalny]]))[0,0]

pacjent2_data_dict = {feature: 0 for feature in final_selected_feature_names}
pacjent2_data_dict.update({
    'Age_at_diagnosis': wiek_pacjent2_znormalizowany,
    'Race': 0, # white
    'IDH1': 0,
    'EGFR': 1,
    'PTEN': 1
})

df_pacjent1 = pd.DataFrame([pacjent1_data_dict], columns=final_selected_feature_names)
df_pacjent2 = pd.DataFrame([pacjent2_data_dict], columns=final_selected_feature_names)

print("\nSztuczne dane pacjentów przygotowane.")
print("Pacjent 1 (dane wejściowe do modelu):")
print(df_pacjent1)
print("\nPacjent 2 (dane wejściowe do modelu):")
print(df_pacjent2)


def predict_ensemble_Ens13(X_new):
    probas_lr = final_lr.predict_proba(X_new)
    probas_svm = final_svm.predict_proba(X_new)
    probas_rf = final_rf_base.predict_proba(X_new)
    probas_ada = final_ada.predict_proba(X_new)

    avg_probas = np.mean([probas_lr, probas_svm, probas_rf, probas_ada], axis=0)
    preds_class = np.argmax(avg_probas, axis=1)
    return preds_class, avg_probas

pred_pacjent1_class, probas_pacjent1 = predict_ensemble_Ens13(df_pacjent1)
pred_pacjent2_class, probas_pacjent2 = predict_ensemble_Ens13(df_pacjent2)

print(f"\n--- Predykcje dla sztucznych pacjentów (Model Ens13_LR_SVM_RF_ADA) ---")
print(f"Pacjent 1: Przewidziana klasa: {pred_pacjent1_class[0]} (0:LGG, 1:GBM), Prawdopodobieństwa: LGG={probas_pacjent1[0,0]:.4f}, GBM={probas_pacjent1[0,1]:.4f}")
print(f"Pacjent 2: Przewidziana klasa: {pred_pacjent2_class[0]} (0:LGG, 1:GBM), Prawdopodobieństwa: LGG={probas_pacjent2[0,0]:.4f}, GBM={probas_pacjent2[0,1]:.4f}")