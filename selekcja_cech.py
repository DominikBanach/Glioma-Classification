from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from collections import Counter

glioma_grading_clinical_and_mutation_features = fetch_ucirepo(id=759)
X_initial = glioma_grading_clinical_and_mutation_features.data.features.copy()
y_initial = glioma_grading_clinical_and_mutation_features.data.targets['Grade'].copy()

print("--- Rozpoczęcie procesu walidacji krzyżowej i kompletnej selekcji cech ---")

N_SPLITS = 10
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)

all_folds_iv_selected_features_lists = {}
all_folds_iv_scores_dict = {}
all_folds_rfe_selected_features = {}
all_folds_rf_selected_features = {}
all_folds_lasso_selected_features = {}
all_folds_voted_features = {}

IV_THRESHOLD = 0.02
N_FEATURES_TO_SELECT_RFE = 12
RF_IMPORTANCE_THRESHOLD_MULTIPLIER = 0.5
LASSO_C_PARAM = 0.1

for fold, (train_idx, val_idx) in enumerate(skf.split(X_initial, y_initial)):
    print(f"\n--- Fałda {fold+1}/{N_SPLITS} ---")
    X_train_fold, X_val_fold = X_initial.iloc[train_idx].copy(), X_initial.iloc[val_idx].copy()
    y_train_fold, y_val_fold = y_initial.iloc[train_idx].copy(), y_initial.iloc[val_idx].copy()

    race_mapping = {
        'white': 0, 'black or african american': 1, 'asian': 2, 'american indian or alaska native': 3
    }
    X_train_fold['Race'] = X_train_fold['Race'].map(race_mapping).fillna(-1)
    X_val_fold['Race'] = X_val_fold['Race'].map(race_mapping).fillna(-1)
    scaler = StandardScaler()
    age_diag_col = 'Age_at_diagnosis'
    X_train_fold[age_diag_col] = scaler.fit_transform(X_train_fold[[age_diag_col]])
    X_val_fold[age_diag_col] = scaler.transform(X_val_fold[[age_diag_col]])

    current_fold_iv_selected = []
    feature_iv_scores_this_fold = {}
    
    train_data_for_woe = pd.concat([X_train_fold, y_train_fold], axis=1)
    target_column_name = y_train_fold.name
    total_pos = train_data_for_woe[target_column_name].sum()
    total_neg = len(train_data_for_woe) - total_pos
    if total_pos > 0 and total_neg > 0:
        X_train_fold_discrete_age = X_train_fold.copy()
        age_bins_col_name = 'Age_bins'
        try:
            X_train_fold_discrete_age[age_bins_col_name] = pd.qcut(X_train_fold_discrete_age[age_diag_col], q=5, duplicates='drop', labels=False)
            temp_train_data_for_woe = pd.concat([X_train_fold_discrete_age, y_train_fold], axis=1)
        except ValueError:
            temp_train_data_for_woe = pd.concat([X_train_fold, y_train_fold], axis=1)
            if age_diag_col not in feature_iv_scores_this_fold: feature_iv_scores_this_fold[age_diag_col] = 0
        for feature_name in X_train_fold.columns:
            current_feature_for_groupby = feature_name
            if feature_name == age_diag_col and age_bins_col_name in X_train_fold_discrete_age.columns:
                current_feature_for_groupby = age_bins_col_name
            iv_feature = 0
            try:
                df_woe_iv = temp_train_data_for_woe.groupby(current_feature_for_groupby, observed=True).agg(
                    pos_count=(target_column_name, lambda x: (x == 1).sum()),
                    neg_count=(target_column_name, lambda x: (x == 0).sum())
                ).reset_index()
                if df_woe_iv.empty:
                    feature_iv_scores_this_fold[feature_name] = 0
                    continue
                df_woe_iv['pos_distr'] = (df_woe_iv['pos_count'] + 0.5) / (total_pos + 0.5 * df_woe_iv.shape[0])
                df_woe_iv['neg_distr'] = (df_woe_iv['neg_count'] + 0.5) / (total_neg + 0.5 * df_woe_iv.shape[0])
                df_woe_iv['woe'] = np.log(df_woe_iv['pos_distr'] / df_woe_iv['neg_distr'])
                df_woe_iv['woe'] = df_woe_iv['woe'].replace([np.inf, -np.inf], 0)
                df_woe_iv['iv_component'] = (df_woe_iv['pos_distr'] - df_woe_iv['neg_distr']) * df_woe_iv['woe']
                iv_feature = df_woe_iv['iv_component'].sum()
                feature_iv_scores_this_fold[feature_name] = iv_feature
                if iv_feature >= IV_THRESHOLD:
                    current_fold_iv_selected.append(feature_name)
            except Exception:
                feature_iv_scores_this_fold[feature_name] = 0
    else:
        for col in X_train_fold.columns:
            feature_iv_scores_this_fold[col] = 0
    all_folds_iv_selected_features_lists[fold] = current_fold_iv_selected
    all_folds_iv_scores_dict[fold] = feature_iv_scores_this_fold

    model_rfe = LogisticRegression(solver='liblinear', random_state=0, max_iter=200)
    n_rfe = min(N_FEATURES_TO_SELECT_RFE, X_train_fold.shape[1])
    rfe_selector = RFE(estimator=model_rfe, n_features_to_select=n_rfe, step=1)
    try:
        rfe_selector.fit(X_train_fold, y_train_fold)
        all_folds_rfe_selected_features[fold] = X_train_fold.columns[rfe_selector.support_].tolist()
    except Exception:
        all_folds_rfe_selected_features[fold] = []

    rf_model = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    try:
        rf_model.fit(X_train_fold, y_train_fold)
        importances = rf_model.feature_importances_
        threshold = np.mean(importances) * RF_IMPORTANCE_THRESHOLD_MULTIPLIER
        all_folds_rf_selected_features[fold] = X_train_fold.columns[importances > threshold].tolist()
    except Exception:
        all_folds_rf_selected_features[fold] = []

    lasso_model = LogisticRegression(penalty='l1', C=LASSO_C_PARAM, solver='liblinear', random_state=0, max_iter=200)
    try:
        lasso_model.fit(X_train_fold, y_train_fold)
        coefficients = lasso_model.coef_[0]
        all_folds_lasso_selected_features[fold] = X_train_fold.columns[np.abs(coefficients) > 1e-5].tolist()
    except Exception:
        all_folds_lasso_selected_features[fold] = []

    all_selected_in_fold = set(all_folds_iv_selected_features_lists.get(fold, [])) | \
                           set(all_folds_rfe_selected_features.get(fold, [])) | \
                           set(all_folds_rf_selected_features.get(fold, [])) | \
                           set(all_folds_lasso_selected_features.get(fold, []))
    all_folds_voted_features[fold] = list(all_selected_in_fold)
    
    if fold == 0:
        print(f"IV wybrało ({len(all_folds_iv_selected_features_lists.get(0,[]))}) cech.")
        print(f"RFE wybrało ({len(all_folds_rfe_selected_features.get(0,[]))}) cech.")
        print(f"RF wybrało ({len(all_folds_rf_selected_features.get(0,[]))}) cech.")
        print(f"LASSO wybrało ({len(all_folds_lasso_selected_features.get(0,[]))}) cech.")
        print(f"PO GŁOSOWANIU wybrano ({len(all_folds_voted_features.get(0,[]))}) cech.")

print("\n--- Koniec Walidacji Krzyżowej i Kompletnej Selekcji Cech ---")

iv_scores_per_feature = {feature: [] for feature in X_initial.columns}
for fold_idx in range(N_SPLITS):
    fold_scores = all_folds_iv_scores_dict.get(fold_idx, {})
    for feature in X_initial.columns:
        iv_scores_per_feature[feature].append(fold_scores.get(feature, 0))
mean_iv_scores_list = []
for feature, scores in iv_scores_per_feature.items():
    mean_iv_scores_list.append({'Cecha': feature, 'Średnie IV': np.mean(scores)})
mean_iv_df = pd.DataFrame(mean_iv_scores_list).sort_values(by='Średnie IV', ascending=False)

latex_mean_iv_table = mean_iv_df.to_latex(
    index=False,
    float_format="%.4f",
    caption="Średnie wartości Information Value (IV) dla cech (po 10 fałdach).",
    label="tab:mean_iv_scores",
    position="H"
)
print("\n\n--- Kod LaTeX dla tabeli średnich IV ---")
print(latex_mean_iv_table)

avg_selected_counts = {
    'Metoda IV (próg > ' + str(IV_THRESHOLD) + ')': np.mean([len(v) for v in all_folds_iv_selected_features_lists.values()]),
    'Metoda RFE (top ' + str(N_FEATURES_TO_SELECT_RFE) + ')': np.mean([len(v) for v in all_folds_rfe_selected_features.values()]),
    'Metoda Random Forest (próg > ' + str(RF_IMPORTANCE_THRESHOLD_MULTIPLIER) + '*średnia)': np.mean([len(v) for v in all_folds_rf_selected_features.values()]),
    'Metoda LASSO (C=' + str(LASSO_C_PARAM) + ')': np.mean([len(v) for v in all_folds_lasso_selected_features.values()]),
    'Po Głosowaniu (min. 1 głos)': np.mean([len(v) for v in all_folds_voted_features.values()])
}
avg_selected_df = pd.DataFrame.from_dict(avg_selected_counts, orient='index', columns=['Średnia liczba wybranych cech'])
avg_selected_df.index.name = 'Metoda Selekcji'

latex_avg_selected_table = avg_selected_df.to_latex(
    float_format="%.2f",
    caption="Średnia liczba cech wybranych przez poszczególne metody selekcji oraz po głosowaniu (po 10 fałdach).",
    label="tab:avg_selected_features",
    position="H"
)
print("\n\n--- Kod LaTeX dla tabeli średniej liczby wybranych cech ---")
print(latex_avg_selected_table)

all_voted_features_flat_list = [feature for fold_features in all_folds_voted_features.values() for feature in fold_features]
feature_vote_counts = Counter(all_voted_features_flat_list)
voted_counts_df = pd.DataFrame(feature_vote_counts.most_common(), columns=['Cecha', 'Liczba fałd (z 10), w których wybrano'])

latex_voted_counts_table = voted_counts_df.to_latex(
    index=False,
    caption="Częstość wyboru poszczególnych cech w mechanizmie głosowania (po 10 fałdach).",
    label="tab:voted_feature_counts",
    position="H"
)
print("\n\n--- Kod LaTeX dla tabeli częstości wyboru cech (głosowanie) ---")
print(latex_voted_counts_table)