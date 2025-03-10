import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV

def xgboost_training(X, y, config, SEED, selected_features):
    """
    Trains an XGBoost model using stratified K-fold cross-validation and optionally performs grid search.
    
    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        config (dict): Configuration settings (e.g., device type, importance type, grid search flag).
        SEED (int): Random seed for reproducibility.
        selected_features (list): List of selected feature names.
        
    Returns:
        dict: Contains best models, best parameters, predicted probabilities, and indices for each fold.
    """
    # Define the hyperparameter grid for grid search
    gbm_param_grid = {
        'colsample_bytree': [0.6, 0.8],
        'n_estimators': [1000, 2000],
        'max_depth': [8, 10],
        'min_child_weight': [4, 6],
    }

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

    # Fixed hyperparameters
    params = {
        'max_depth': 6,
        'eta': 0.01,
        'objective': 'binary:logistic',
        'nthread': 32,
        'eval_metric': 'auc',
        'scale_pos_weight': 1,
        'min_child_weight': 5,
        'colsample_bytree': 0.6,
        'subsample': 0.8,
        'gamma': 1,
        'n_estimators': 500,
        'verbosity': 1,
        'seed': SEED,
        'enable_categorical': True
    }

    # Initialize lists to store results
    y_pred_proba_list = []
    y_pred_list = []
    y_ind_list = []
    best_models_list = []
    best_params_list = []
    fi_list = []

    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        xgb_clf = xgb.XGBClassifier(**params)  # Fresh classifier for each fold

        print(f"Fold: {fold_idx + 1}")
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]

        # Perform grid search if enabled
        if config["grid_search"]:
            clf = GridSearchCV(
                estimator=xgb_clf,
                param_grid=gbm_param_grid,
                scoring='roc_auc',
                cv=5,
                n_jobs=10,
                verbose=1
            )
        else:
            clf = xgb_clf

        clf.fit(X_train, y_train)

        # Retrieve best model and parameters
        if config["grid_search"]:
            best_model_clf = clf.best_estimator_
            best_params_ = clf.best_params_
        else:
            best_model_clf = clf
            best_params_ = clf.get_params()

        # Feature importance extraction
        m = best_model_clf.get_booster()
        fi = m.get_score(importance_type=config["importance_type"])
        fi_list.append(fi)

        # Predict probabilities and labels
        y_pred_proba = best_model_clf.predict_proba(X_test)[:, 1]  # Probability predictions
        y_pred = best_model_clf.predict(X_test)  # Binary predictions

        # Store results
        y_pred_proba_list.append(y_pred_proba)
        y_pred_list.append(y_pred)
        y_ind_list.append(test_index)
        best_params_list.append(best_params_)
        best_models_list.append(best_model_clf)

    return {
        "best_models": best_models_list,
        "best_params": best_params_list,
        "y_pred_proba": y_pred_proba_list,
        "y_pred": y_pred_list,
        "y_indices": y_ind_list,
        "feature_importance": fi_list
    }