from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

glioma_grading_clinical_and_mutation_features = fetch_ucirepo(id=759)
X_initial = glioma_grading_clinical_and_mutation_features.data.features.copy()
y_initial = glioma_grading_clinical_and_mutation_features.data.targets['Grade'].copy()

print("--- Rozpoczęcie procesu walidacji krzyżowej, selekcji cech, trenowania modeli bazowych i zespołowych ---")

N_SPLITS = 10
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)

all_folds_iv_selected_features_lists = {}
all_folds_iv_scores_dict = {}
all_folds_rfe_selected_features = {}
all_folds_rf_selected_features = {}
all_folds_lasso_selected_features = {}
all_folds_voted_features = {}
all_folds_base_model_probas = {}
all_folds_base_model_metrics = {}
all_folds_ensemble_model_metrics = {}

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
    current_fold_rfe_selected = []
    current_fold_rf_selected = []
    current_fold_lasso_selected = []

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
                df_woe_iv['woe'] = df_woe_iv['woe'].replace([np.inf, -np.inf, np.nan], 0)
                df_woe_iv['iv_component'] = (df_woe_iv['pos_distr'] - df_woe_iv['neg_distr']) * df_woe_iv['woe']
                iv_feature = df_woe_iv['iv_component'].sum()
                feature_iv_scores_this_fold[feature_name] = iv_feature
                if iv_feature >= IV_THRESHOLD: current_fold_iv_selected.append(feature_name)
            except Exception: feature_iv_scores_this_fold[feature_name] = 0
    else:
        for col in X_train_fold.columns: feature_iv_scores_this_fold[col] = 0
    all_folds_iv_selected_features_lists[fold] = current_fold_iv_selected
    all_folds_iv_scores_dict[fold] = feature_iv_scores_this_fold

    model_rfe = LogisticRegression(solver='liblinear', random_state=0, max_iter=200)
    n_rfe = min(N_FEATURES_TO_SELECT_RFE, X_train_fold.shape[1])
    if n_rfe > 0:
        rfe_selector = RFE(estimator=model_rfe, n_features_to_select=n_rfe, step=1)
        try:
            rfe_selector.fit(X_train_fold, y_train_fold)
            current_fold_rfe_selected = X_train_fold.columns[rfe_selector.support_].tolist()
        except Exception: current_fold_rfe_selected = []
    else: current_fold_rfe_selected = []
    all_folds_rfe_selected_features[fold] = current_fold_rfe_selected

    rf_importance_model = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    try:
        rf_importance_model.fit(X_train_fold, y_train_fold)
        importances = rf_importance_model.feature_importances_
        if len(importances) > 0:
            threshold = np.mean(importances) * RF_IMPORTANCE_THRESHOLD_MULTIPLIER
            current_fold_rf_selected = X_train_fold.columns[importances > threshold].tolist()
        else: current_fold_rf_selected = []
    except Exception: current_fold_rf_selected = []
    all_folds_rf_selected_features[fold] = current_fold_rf_selected

    lasso_model = LogisticRegression(penalty='l1', C=LASSO_C_PARAM, solver='liblinear', random_state=0, max_iter=200)
    try:
        lasso_model.fit(X_train_fold, y_train_fold)
        coefficients = lasso_model.coef_[0]
        current_fold_lasso_selected = X_train_fold.columns[np.abs(coefficients) > 1e-5].tolist()
    except Exception: current_fold_lasso_selected = []
    all_folds_lasso_selected_features[fold] = current_fold_lasso_selected

    all_selected_in_fold = set(current_fold_iv_selected) | \
                           set(current_fold_rfe_selected) | \
                           set(current_fold_rf_selected) | \
                           set(current_fold_lasso_selected)
    voted_features_this_fold = list(all_selected_in_fold)
    all_folds_voted_features[fold] = voted_features_this_fold

    if not voted_features_this_fold:
        print(f"Ostrzeżenie w Fałdzie {fold+1}: Brak cech po głosowaniu. Używam wszystkich cech dla modeli bazowych.")
        X_train_selected = X_train_fold.copy(); X_val_selected = X_val_fold.copy()
    else:
        X_train_selected = X_train_fold[voted_features_this_fold]; X_val_selected = X_val_fold[voted_features_this_fold]

    if X_train_selected.empty or X_val_selected.empty:
        print(f"Ostrzeżenie w Fałdzie {fold+1}: Pusty zbiór danych po selekcji cech. Pomijanie trenowania modeli bazowych i zespołowych.")
        all_folds_base_model_probas[fold] = {}; all_folds_base_model_metrics[fold] = {}; all_folds_ensemble_model_metrics[fold] = {}
        continue
        
    print(f"--- Trenowanie Modeli Bazowych i Obliczanie Metryk na {X_train_selected.shape[1]} cechach w Fałdzie {fold+1} ---")
    fold_base_model_probas_current_fold = {}
    fold_base_model_metrics_current_fold = {}
    trained_models_current_fold = {}

    models_to_train_dict = {
        "LR": LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=0, solver='liblinear', max_iter=200),
        "SVM": SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=0),
        "KNN": KNeighborsClassifier(n_neighbors=5, metric='minkowski'),
        "RF_base": RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=0, n_jobs=-1),
        "ADA": AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=0)
    }

    if fold == 0: plt.figure(figsize=(10, 8))

    for model_name, model_instance in models_to_train_dict.items():
        try:
            model_instance.fit(X_train_selected, y_train_fold)
            trained_models_current_fold[model_name] = model_instance
            probas = model_instance.predict_proba(X_val_selected)
            preds = np.argmax(probas, axis=1)
            fold_base_model_probas_current_fold[model_name] = probas

            acc = accuracy_score(y_val_fold, preds)
            auc = roc_auc_score(y_val_fold, probas[:, 1])
            f1 = f1_score(y_val_fold, preds, zero_division=0)
            prec = precision_score(y_val_fold, preds, zero_division=0)
            rec = recall_score(y_val_fold, preds, zero_division=0)
            cm = confusion_matrix(y_val_fold, preds)
            
            if cm.shape == (2,2): tn, fp, fn, tp = cm.ravel()
            elif cm.shape == (1,1) and y_val_fold.nunique() == 1 and y_val_fold.unique()[0] == 0: tn, fp, fn, tp = cm[0,0], 0, 0, 0
            elif cm.shape == (1,1) and y_val_fold.nunique() == 1 and y_val_fold.unique()[0] == 1: tn, fp, fn, tp = 0, 0, 0, cm[0,0]
            else: tn, fp, fn, tp = 0,0,0,0
            
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            fold_base_model_metrics_current_fold[model_name] = {'ACC': acc, 'AUC': auc, 'F1': f1, 'PRE': prec, 'REC': rec, 'SPEC': spec, 'CM': cm}
            
            if fold == 0:
                fpr, tpr, _ = roc_curve(y_val_fold, probas[:, 1])
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')
        except Exception as e:
            print(f"Błąd {model_name} (bazowy) w Fałdzie {fold+1}: {e}")
            fold_base_model_probas_current_fold[model_name] = np.zeros((X_val_selected.shape[0], 2))
            fold_base_model_metrics_current_fold[model_name] = {m: 0 for m in ['ACC', 'AUC', 'F1', 'PRE', 'REC', 'SPEC', 'CM']}
    
    if fold == 0:
        plt.plot([0, 1], [0, 1], color='grey', linestyle='--'); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title(f'Krzywe ROC Modeli Bazowych (Fałda {fold+1})')
        plt.legend(loc="lower right"); plt.savefig("krzywe_roc_fala_1.png", dpi=300); plt.show()

        if 'RF_base' in trained_models_current_fold:
            rf_model_for_tree = trained_models_current_fold['RF_base']
            if hasattr(rf_model_for_tree, 'estimators_') and len(rf_model_for_tree.estimators_) > 0:
                single_tree = rf_model_for_tree.estimators_[0]
                dot_data = export_graphviz(single_tree, out_file=None, feature_names=X_train_selected.columns.tolist(), class_names=['LGG (0)', 'GBM (1)'], filled=True, rounded=True, special_characters=True, impurity=True, proportion=True)
                with open("przykładowe_drzewo_rf.dot", "w") as f: f.write(dot_data)
                print("\nZapisano 'przykładowe_drzewo_rf.dot'. Użyj: dot -Tpng przykładowe_drzewo_rf.dot -o przykładowe_drzewo_rf.png")
        
        plt.figure(figsize=(18, 12)); plot_idx = 1
        for model_name_cm, metrics_dict_cm in fold_base_model_metrics_current_fold.items():
            if 'CM' in metrics_dict_cm and isinstance(metrics_dict_cm['CM'], np.ndarray):
                if plot_idx <= 6 :
                    plt.subplot(2, 3, plot_idx)
                    sns.heatmap(metrics_dict_cm['CM'], annot=True, fmt="d", cmap="Blues", xticklabels=['Przew. LGG', 'Przew. GBM'], yticklabels=['Rzecz. LGG', 'Rzecz. GBM'])
                    plt.title(f'Macierz Pomyłek - {model_name_cm}'); plt.ylabel('Rzeczywista'); plt.xlabel('Przewidziana')
                    plot_idx += 1
        if plot_idx > 1:
            plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.suptitle(f"Macierze Pomyłek Modeli Bazowych (Fałda {fold+1})", fontsize=16)
            plt.savefig("macierze_pomylek_fala_1.png", dpi=300); plt.show()

    all_folds_base_model_probas[fold] = fold_base_model_probas_current_fold
    all_folds_base_model_metrics[fold] = fold_base_model_metrics_current_fold

    print(f"--- Modele Zespołowe (Soft Voting) w Fałdzie {fold+1} ---")
    ensemble_model_combinations = [
        ("Ens1_LR_SVM_KNN", ["LR", "SVM", "KNN"]), ("Ens2_LR_SVM_RF", ["LR", "SVM", "RF_base"]), ("Ens3_LR_SVM_ADA", ["LR", "SVM", "ADA"]),
        ("Ens4_LR_KNN_RF", ["LR", "KNN", "RF_base"]), ("Ens5_LR_KNN_ADA", ["LR", "KNN", "ADA"]), ("Ens6_LR_RF_ADA", ["LR", "RF_base", "ADA"]),
        ("Ens7_SVM_KNN_RF", ["SVM", "KNN", "RF_base"]), ("Ens8_SVM_KNN_ADA", ["SVM", "KNN", "ADA"]), ("Ens9_SVM_RF_ADA", ["SVM", "RF_base", "ADA"]),
        ("Ens10_KNN_RF_ADA", ["KNN", "RF_base", "ADA"]), ("Ens11_LR_SVM_KNN_RF", ["LR", "SVM", "KNN", "RF_base"]),
        ("Ens12_LR_SVM_KNN_ADA", ["LR", "SVM", "KNN", "ADA"]), ("Ens13_LR_SVM_RF_ADA", ["LR", "SVM", "RF_base", "ADA"]),
        ("Ens14_LR_KNN_RF_ADA", ["LR", "KNN", "RF_base", "ADA"]), ("Ens15_SVM_KNN_RF_ADA", ["SVM", "KNN", "RF_base", "ADA"]),
        ("Ens16_LR_SVM_KNN_RF_ADA", ["LR", "SVM", "KNN", "RF_base", "ADA"])
    ]
    fold_ensemble_model_metrics = {}
    available_model_probas_keys = fold_base_model_probas_current_fold.keys()
    for ensemble_name, model_keys_in_ensemble in ensemble_model_combinations:
        if not all(key in available_model_probas_keys for key in model_keys_in_ensemble):
            fold_ensemble_model_metrics[ensemble_name] = {m: 0 for m in metric_keys}; continue
        probas_for_ensemble = [fold_base_model_probas_current_fold[key] for key in model_keys_in_ensemble if fold_base_model_probas_current_fold.get(key) is not None and fold_base_model_probas_current_fold.get(key).size > 0]
        if not probas_for_ensemble or not all(p.shape == probas_for_ensemble[0].shape for p in probas_for_ensemble):
            fold_ensemble_model_metrics[ensemble_name] = {m: 0 for m in metric_keys}; continue
        avg_probas = np.mean(probas_for_ensemble, axis=0)
        ensemble_preds = np.argmax(avg_probas, axis=1)
        try:
            acc = accuracy_score(y_val_fold, ensemble_preds); auc = roc_auc_score(y_val_fold, avg_probas[:, 1]); f1 = f1_score(y_val_fold, ensemble_preds, zero_division=0)
            prec = precision_score(y_val_fold, ensemble_preds, zero_division=0); rec = recall_score(y_val_fold, ensemble_preds, zero_division=0)
            cm_ens = confusion_matrix(y_val_fold, ensemble_preds)
            if cm_ens.shape == (2,2): tn_ens, fp_ens, fn_ens, tp_ens = cm_ens.ravel()
            elif cm_ens.shape == (1,1) and y_val_fold.nunique() == 1 and y_val_fold.unique()[0] == 0: tn_ens,fp_ens,fn_ens,tp_ens = cm_ens[0,0],0,0,0
            elif cm_ens.shape == (1,1) and y_val_fold.nunique() == 1 and y_val_fold.unique()[0] == 1: tn_ens,fp_ens,fn_ens,tp_ens = 0,0,0,cm_ens[0,0]
            else: tn_ens,fp_ens,fn_ens,tp_ens = 0,0,0,0
            spec = tn_ens / (tn_ens + fp_ens) if (tn_ens + fp_ens) > 0 else 0
            fold_ensemble_model_metrics[ensemble_name] = {'ACC': acc, 'AUC': auc, 'F1': f1, 'PRE': prec, 'REC': rec, 'SPEC': spec, 'CM': cm_ens}
        except Exception as e:
            print(f"Błąd {ensemble_name} (zespół) w Fałdzie {fold+1}: {e}")
            fold_ensemble_model_metrics[ensemble_name] = {m: 0 for m in ['ACC', 'AUC', 'F1', 'PRE', 'REC', 'SPEC', 'CM']}
    all_folds_ensemble_model_metrics[fold] = fold_ensemble_model_metrics

print("\n--- Koniec Procesu Walidacji Krzyżowej ---")

iv_scores_per_feature = {feature: [] for feature in X_initial.columns}
for fold_idx in range(N_SPLITS):
    fold_scores = all_folds_iv_scores_dict.get(fold_idx, {})
    for feature in X_initial.columns:
        iv_scores_per_feature[feature].append(fold_scores.get(feature, 0))
mean_iv_scores_list = []
for feature, scores in iv_scores_per_feature.items():
    mean_iv_scores_list.append({'Cecha': feature, 'Średnie IV': np.mean(scores) if scores else 0})
mean_iv_df = pd.DataFrame(mean_iv_scores_list).sort_values(by='Średnie IV', ascending=False)
latex_mean_iv_table = mean_iv_df.to_latex(index=False, float_format="%.4f", caption="Średnie wartości Information Value (IV) dla cech (po 10 fałdach).", label="tab:mean_iv_scores", position="H")
print("\n\n--- Kod LaTeX dla tabeli średnich IV ---")
print(latex_mean_iv_table)

avg_selected_counts = {
    f'Metoda IV (próg > {IV_THRESHOLD})': np.mean([len(v) for v in all_folds_iv_selected_features_lists.values()] if all_folds_iv_selected_features_lists else [0]),
    f'Metoda RFE (top {N_FEATURES_TO_SELECT_RFE})': np.mean([len(v) for v in all_folds_rfe_selected_features.values()] if all_folds_rfe_selected_features else [0]),
    f'Metoda Random Forest (próg > {RF_IMPORTANCE_THRESHOLD_MULTIPLIER}*średnia)': np.mean([len(v) for v in all_folds_rf_selected_features.values()] if all_folds_rf_selected_features else [0]),
    f'Metoda LASSO (C={LASSO_C_PARAM})': np.mean([len(v) for v in all_folds_lasso_selected_features.values()] if all_folds_lasso_selected_features else [0]),
    'Po Głosowaniu (min. 1 głos)': np.mean([len(v) for v in all_folds_voted_features.values()] if all_folds_voted_features else [0])
}
avg_selected_df = pd.DataFrame.from_dict(avg_selected_counts, orient='index', columns=['Średnia liczba wybranych cech'])
avg_selected_df.index.name = 'Metoda Selekcji'
latex_avg_selected_table = avg_selected_df.to_latex(float_format="%.2f", caption="Średnia liczba cech wybranych przez poszczególne metody selekcji oraz po głosowaniu (po 10 fałdach).", label="tab:avg_selected_features", position="H")
print("\n\n--- Kod LaTeX dla tabeli średniej liczby wybranych cech ---")
print(latex_avg_selected_table)

all_voted_features_flat_list = [feature for fold_features in all_folds_voted_features.values() for feature in fold_features]
feature_vote_counts = Counter(all_voted_features_flat_list)
voted_counts_df = pd.DataFrame(feature_vote_counts.most_common(), columns=['Cecha', 'Liczba fałd (z 10), w których wybrano'])
latex_voted_counts_table = voted_counts_df.to_latex(index=False, caption="Częstość wyboru poszczególnych cech w mechanizmie głosowania (po 10 fałdach).", label="tab:voted_feature_counts", position="H")
print("\n\n--- Kod LaTeX dla tabeli częstości wyboru cech (głosowanie) ---")
print(latex_voted_counts_table)

metrics_summary = []
model_names_list = ["LR", "SVM", "KNN", "RF_base", "ADA"]
metric_keys = ['ACC', 'AUC', 'F1', 'PRE', 'REC', 'SPEC']
for model_name in model_names_list:
    model_metrics_all_folds = {key: [] for key in metric_keys}
    for fold_idx in range(N_SPLITS):
        fold_metrics = all_folds_base_model_metrics.get(fold_idx, {}).get(model_name, {m: 0 for m in metric_keys})
        for key in metric_keys: model_metrics_all_folds[key].append(fold_metrics.get(key, 0))
    summary_row = {'Model': model_name}
    for key in metric_keys:
        mean_val = np.mean(model_metrics_all_folds[key]); std_val = np.std(model_metrics_all_folds[key])
        summary_row[f'{key} (Śr. ± Std)'] = f"{mean_val:.3f} ± {std_val:.3f}"; summary_row[f'{key}_mean'] = mean_val
    metrics_summary.append(summary_row)
metrics_summary_df = pd.DataFrame(metrics_summary).sort_values(by='ACC_mean', ascending=False)
latex_table_cols = ['Model'] + [f'{key} (Śr. ± Std)' for key in metric_keys]
metrics_summary_df_for_latex = metrics_summary_df[latex_table_cols]
latex_base_model_metrics_table = metrics_summary_df_for_latex.to_latex(index=False, caption="Średnie metryki wydajności dla indywidualnych modeli bazowych (po 10 fałdach walidacji krzyżowej). Wartości przedstawiono jako 'średnia ± odchylenie standardowe'.", label="tab:base_model_metrics", position="H", column_format='l' + 'c' * len(metric_keys))
print("\n\n--- Kod LaTeX dla tabeli metryk modeli bazowych ---")
print(latex_base_model_metrics_table)

ensemble_metrics_summary_list = []
ensemble_names_for_summary = [name for name, _ in ensemble_model_combinations]
for ens_name in ensemble_names_for_summary:
    ens_metrics_all_folds = {key: [] for key in metric_keys}
    for fold_idx in range(N_SPLITS):
        fold_ens_metrics = all_folds_ensemble_model_metrics.get(fold_idx, {}).get(ens_name, {m: 0 for m in metric_keys})
        for key in metric_keys: ens_metrics_all_folds[key].append(fold_ens_metrics.get(key, 0))
    summary_row = {'Model Zespołowy': ens_name}
    for key in metric_keys:
        mean_val = np.mean(ens_metrics_all_folds[key]); std_val = np.std(ens_metrics_all_folds[key])
        summary_row[f'{key} (Śr. ± Std)'] = f"{mean_val:.3f} ± {std_val:.3f}"; summary_row[f'{key}_mean_ens'] = mean_val
    ensemble_metrics_summary_list.append(summary_row)
ensemble_metrics_summary_df = pd.DataFrame(ensemble_metrics_summary_list).sort_values(by='ACC_mean_ens', ascending=False)
latex_ensemble_table_cols = ['Model Zespołowy'] + [f'{key} (Śr. ± Std)' for key in metric_keys]
ensemble_metrics_summary_df_for_latex = ensemble_metrics_summary_df[latex_ensemble_table_cols]
latex_ensemble_model_metrics_table = ensemble_metrics_summary_df_for_latex.to_latex(index=False, caption="Średnie metryki wydajności dla modeli zespołowych (soft voting) po 10 fałdach walidacji krzyżowej. Wartości przedstawiono jako 'średnia ± odchylenie standardowe'.", label="tab:ensemble_model_metrics", position="H", column_format='l' + 'c' * len(metric_keys))
print("\n\n--- Kod LaTeX dla tabeli metryk modeli ZESPOŁOWYCH ---")
print(latex_ensemble_model_metrics_table)