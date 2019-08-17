import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")



X_train_scaled = pd.read_csv('./EarthquakePrediction/scaled_train_X.csv',delimiter=',')
train_y = pd.read_csv('./EarthquakePrediction/train_y.csv',delimiter=',')
X_train_scaled=X_train_scaled.iloc[:,2:-1]
train_y=train_y.iloc[:,-1]
submission = pd.read_csv('./sample_submission.csv', index_col='seg_id')
scaled_test_X = pd.DataFrame(columns=X_train_scaled.columns, dtype=np.float64,
                      index=submission.index)
X_test_scaled = pd.read_csv('X_test_scaled.csv',delimiter=',',header=None)
X_test_scaled=X_test_scaled.iloc[:,:-1]

dim=X_train_scaled.shape[1]
dim_out=3
epochs=150
bias=15
batch_size=250;

n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)
train_columns = X_train_scaled.columns.values

params = {'num_leaves': 51,
         'min_data_in_leaf': 10, 
         'objective':'regression',
         'max_depth': -1,
         'learning_rate': 0.001,
         "boosting": "gbdt",
         "feature_fraction": 0.91,
         "bagging_freq": 1,
         "bagging_fraction": 0.91,
         "bagging_seed": 42,
         "metric": 'mae',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "nthread": -1,
         "random_state": 42}



oof = np.zeros(len(X_train_scaled))
predictions = np.zeros(len(scaled_test_X))
feature_importance_df = pd.DataFrame()
#run model
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train_scaled,train_y.values)):
    strLog = "fold {}".format(fold_)
    print(strLog)
    
    X_tr, X_val = X_train_scaled.iloc[trn_idx], X_train_scaled.iloc[val_idx]
    y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]
    
    model = lgb.LGBMRegressor(**params, n_estimators = 2000, n_jobs = -1)
    model.fit(X_tr, y_tr)
    
    oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration_)
    #feature importance
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = train_columns
    fold_importance_df["importance"] = model.feature_importances_[:len(train_columns)]
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    #predictions
    predictions += model.predict(scaled_test_X, num_iteration=model.best_iteration_) / folds.n_splits

cols = (feature_importance_df[["Feature", "importance"]]
        .groupby("Feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:200].index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(14,26))
sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('LightGBM Features (averaged over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')
